#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "cilent/image_processor.h"
#include "cilent/kalman_filter.h"
#include "cilent/serial_sender.h"

namespace cilent {

class AutoAimNode : public ImageProcessorNode {
 public:
	explicit AutoAimNode(const rclcpp::NodeOptions &options =
													 rclcpp::NodeOptions())
		: ImageProcessorNode(options),
			serial_device_(declare_parameter<std::string>("serial_device",
									 "/dev/pts/8")),  // 串口设备路径；改成实际串口(如 /dev/ttyUSB0)，错了会导致无法发云台/开火指令。
			sender_(serial_device_),
			bullet_speed_px_s_(declare_parameter<double>("bullet_speed_px_s", 600.0)),  // 子弹等效像素速度(px/s)；调大提前量变小，调小提前量变大。
			enemy_speed_max_px_s_(
				declare_parameter<double>("enemy_speed_max_px_s", 300.0)),  // 预留参数(当前未生效)；建议暂不改。
			track_match_px_(declare_parameter<double>("track_match_px", 90.0)),  // 轨迹匹配门限(px)；调大更容易把近邻目标并成同一轨迹，调小更容易新建轨迹。
			target_commit_ms_(declare_parameter<int>("target_commit_ms", 220)),  // 预留参数(当前未生效)；建议暂不改。
			hold_lost_grace_ms_(declare_parameter<int>("hold_lost_grace_ms", 260)),  // 丢目标后保留锁定时长(ms)；调大抗瞬时丢帧更强，调小停火更快。
			switch_score_margin_(declare_parameter<double>("switch_score_margin", 0.9)),  // 预留参数(当前未生效)；建议暂不改。
			switch_min_interval_ms_(
				declare_parameter<int>("switch_min_interval_ms", 180)),  // 预留参数(当前未生效)；建议暂不改。
			yaw_move_penalty_w_(declare_parameter<double>("yaw_move_penalty_w", 1.1)),  // 预留参数(当前未生效)；建议暂不改。
			center_bias_w_(declare_parameter<double>("center_bias_w", 0.25)),  // 预留参数(当前未生效)；建议暂不改。
			yaw_fov_deg_(declare_parameter<double>("yaw_fov_deg", 90.0)),  // 像素到yaw映射的视场角；调大同样像素偏差会输出更大yaw。
			yaw_sign_(declare_parameter<double>("yaw_sign", -1.0)),  // yaw方向符号；方向反了改成 1.0。
			yaw_offset_deg_(declare_parameter<double>("yaw_offset_deg", 90.0)),  // yaw零点偏置；用于对齐机械中位，调大整体向一侧偏。
			yaw_send_interval_ms_(declare_parameter<int>("yaw_send_interval_ms", 120)),  // yaw发送周期(ms)；调小更跟手但串口负担更高。
			fire_x_tol_px_(declare_parameter<int>("fire_x_tol_px", 60)),  // 预留参数(当前未生效)；建议暂不改。
			fire_cooldown_ms_(declare_parameter<int>("fire_cooldown_ms", 50)),  // 开火最小间隔(ms)；调小射速更快，调大更保守。
			edge_ignore_px_(declare_parameter<int>("edge_ignore_px", 20)),  // 画面边缘禁射区(px)；调大更安全但会少打边缘目标。
			lead_time_s_(declare_parameter<double>("lead_time_s", 0.0)),  // 固定延迟补偿(s)；调大整体提前增加，调小整体提前减少。
			lead_y_comp_start_ratio_(
				declare_parameter<double>("lead_y_comp_start_ratio", 0.65)),  // 从该y比例开始抑制提前量(0顶端,1底部)；调小更早介入抑制。
			lead_y_comp_scale_bottom_(
				declare_parameter<double>("lead_y_comp_scale_bottom", 0.45)),  // 底部保留比例；调小底部抑制更强，调大更接近原始预测。
			lead_y_comp_fixed_px_bottom_(
				declare_parameter<double>("lead_y_comp_fixed_px_bottom", 80.0)),  // 底部固定回拉(px)；调大可抑制过前，但过大可能变过后。
			lead_center_window_ratio_(
				declare_parameter<double>("lead_center_window_ratio", 0.22)),  // 中心抑制窗口占半屏比例；调大中心附近抑制范围更宽。
			lead_center_scale_min_(
				declare_parameter<double>("lead_center_scale_min", 0.3)),  // 中心最小保留比例；调小中心更稳但跟随更慢。
			kf_sigma_a_(declare_parameter<double>("kf_sigma_a", 80.0)),  // KF过程噪声(加速度)；调大更灵敏，调小更平滑。
			kf_sigma_z_(declare_parameter<double>("kf_sigma_z", 0.1)),  // KF测量噪声；调大更依赖预测，调小更跟随检测框。
			target_reset_ms_(declare_parameter<int>("target_reset_ms", 100)),  // 丢目标后恢复开火等待(ms)；调大更谨慎，调小恢复更快。
			auto_fire_(declare_parameter<bool>("auto_fire", true)),  // 自动开火开关；false 时只转向不发火。
			last_fire_time_(0, 0, RCL_ROS_TIME),
			last_target_lost_time_(0, 0, RCL_ROS_TIME),
			last_yaw_send_time_(0, 0, RCL_STEADY_TIME),
			stop_firing_(false) {
		if (!sender_.open()) {
			RCLCPP_WARN(get_logger(), "Failed to open serial device %s",
								serial_device_.c_str());
		}

		const int fire_period_ms = std::max(1, fire_cooldown_ms_);
		fire_timer_ = create_wall_timer(
			std::chrono::milliseconds(fire_period_ms),
			std::bind(&AutoAimNode::onFireTimer, this));
	}

 protected:
	void onImage(const sensor_msgs::msg::Image::SharedPtr msg) override {
		ImageContext ctx;
		if (!buildImageContext(msg, ctx)) {
			return;
		}

		const cv::Mat &frame = ctx.cv_ptr->image;
		const cv::Rect &crop = ctx.crop;
		const cv::Vec3d &mean_bgr = ctx.mean_bgr;
		const std::string &own_color = ctx.color;
		const DetectionResult &detection = ctx.detection;

		if (!detection.found) {
			const rclcpp::Time now = this->now();
			if (last_target_lost_time_.nanoseconds() <= 0) {
				last_target_lost_time_ = now;
			}
			const bool lost_too_long =
				(now - last_target_lost_time_).nanoseconds() >
				static_cast<int64_t>(std::max(1, hold_lost_grace_ms_)) * 1000000LL;
			stop_firing_ = true;
			if (lost_too_long) {
				last_selected_track_id_ = -1;
				target_lock_frames_ = 0;
			}
			maybeSaveDebugImages(frame, crop, detection, nullptr);
			RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
											 "Center color: %s (B=%.1f G=%.1f R=%.1f)",
											 own_color.c_str(), mean_bgr[0], mean_bgr[1],
											 mean_bgr[2]);
			return;
		}

		if (stop_firing_) {
			const rclcpp::Time now = this->now();
			const bool reset_ok =
				last_target_lost_time_.nanoseconds() > 0 &&
				(now - last_target_lost_time_).nanoseconds() >=
					static_cast<int64_t>(target_reset_ms_) * 1000000LL;
			if (reset_ok) {
				stop_firing_ = false;
			}
		}

		const rclcpp::Time now = this->now();
		last_target_lost_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
		pruneTracks(now);

		DetectionResult selected = detection;
		int selected_track_id = -1;
		const float x_center = (static_cast<float>(frame.cols) - 1.0f) * 0.5f;
		const double bullet_speed = std::max(1.0, bullet_speed_px_s_);
		const double latency_comp_s = std::max(0.0, lead_time_s_);
		std::vector<cv::Point2f> all_lead_points;
		std::vector<int> all_lead_track_ids;
		std::unordered_map<int, cv::Point2f> lead_point_by_track;
		std::unordered_map<int, cv::Rect> box_by_track;
		all_lead_points.reserve(detection.boxes.size());
		all_lead_track_ids.reserve(detection.boxes.size());
		lead_point_by_track.reserve(detection.boxes.size());
		box_by_track.reserve(detection.boxes.size());
		if (!detection.boxes.empty()) {
			int detector_best_track_id = -1;

			for (const auto &box : detection.boxes) {
				if (box.width <= 0 || box.height <= 0) {
					continue;
				}

				const float cx =
					static_cast<float>(box.x + box.width * 0.5f + crop.x);
				const float cy =
					static_cast<float>(box.y + box.height * 0.5f + crop.y);

				const int track_id = matchOrCreateTrack(cv::Point2f(cx, cy), now);
				TrackState *track = findTrackById(track_id);
				if (!track) {
					continue;
				}

				track->center = cv::Point2f(cx, cy);
				track->bbox = cv::Rect(box.x + crop.x, box.y + crop.y,
											 box.width, box.height);
				track->last_seen = now;
				track->label = detection.label;

				auto [kf_it, inserted] = track_kfs_.emplace(
					track_id, KalmanFilterCV2D(0.02, kf_sigma_a_, kf_sigma_z_));
				if (inserted) {
					kf_it->second.reset(cx, cy, 0.0, 0.0);
					track_kf_last_meas_time_[track_id] = now;
				}

				KalmanFilterCV2D &track_kf = kf_it->second;
				track_kf.setProcessNoise(kf_sigma_a_);
				track_kf.setMeasurementNoise(kf_sigma_z_);

				auto t_it = track_kf_last_meas_time_.find(track_id);
				if (t_it != track_kf_last_meas_time_.end()) {
					const double dt = (now - t_it->second).seconds();
					if (dt > 0.0) {
						track_kf.setDeltaT(std::clamp(dt, 1e-3, 0.2));
						track_kf.predict();
					}
				}
				track_kf.update(cx, cy);
				track_kf_last_meas_time_[track_id] = now;

				const double travel_px =
					std::abs(static_cast<double>(track->center.x) -
							 static_cast<double>(x_center));
				const double flight_time_s = travel_px / bullet_speed;
				const double lead_time_dynamic_s =
					std::clamp(flight_time_s + latency_comp_s, 0.0, 0.5);
				const Eigen::Vector2d lead_xy =
					track_kf.predictAhead(lead_time_dynamic_s);
				const cv::Point2f raw_lead_point(static_cast<float>(lead_xy.x()),
											   static_cast<float>(lead_xy.y()));
				const cv::Point2f lead_point = compensateLeadByY(
					track->center, raw_lead_point, frame.cols, frame.rows);
				all_lead_points.push_back(lead_point);
				all_lead_track_ids.push_back(track_id);
				lead_point_by_track[track_id] = lead_point;
				box_by_track[track_id] = box;
				if (box == detection.bbox) {
					detector_best_track_id = track_id;
				}
			}

			// Keep lock on the previous track when it is still visible to avoid
			// left-right jitter from switching between multiple candidate boxes.
			if (last_selected_track_id_ >= 0 &&
				lead_point_by_track.find(last_selected_track_id_) !=
					lead_point_by_track.end()) {
				selected_track_id = last_selected_track_id_;
			} else if (detector_best_track_id >= 0) {
				selected_track_id = detector_best_track_id;
			} else if (!all_lead_track_ids.empty()) {
				selected_track_id = all_lead_track_ids.front();
			}

			auto selected_box_it = box_by_track.find(selected_track_id);
			if (selected_box_it != box_by_track.end()) {
				const cv::Rect &sel_box = selected_box_it->second;
				selected.bbox = sel_box;
				selected.center = cv::Point2f(
					static_cast<float>(sel_box.x + sel_box.width * 0.5f),
					static_cast<float>(sel_box.y + sel_box.height * 0.5f));
				selected.area = static_cast<double>(sel_box.area());
			}
		}

		const float x = selected.center.x + static_cast<float>(crop.x);
		const float y = selected.center.y + static_cast<float>(crop.y);
		const bool track_switched =
			(selected_track_id >= 0) &&
			(selected_track_id != last_selected_track_id_);
		const int edge = std::max(0, edge_ignore_px_);
		if (x < static_cast<float>(edge) ||
			x > static_cast<float>(frame.cols - 1 - edge)) {
			last_target_lost_time_ = this->now();
			stop_firing_ = true;
			RCLCPP_INFO_THROTTLE(
				get_logger(), *get_clock(), 500,
				"Fire blocked: target at edge x=%.1f edge_ignore_px=%d", x, edge);
			return;
		}
		const float y_lb = static_cast<float>(frame.rows - 1) - y;

		if (track_switched) {
			RCLCPP_INFO_THROTTLE(
				get_logger(), *get_clock(), 300,
				"Track switched (%d->%d)",
				last_selected_track_id_, selected_track_id);
		}

		if (all_lead_points.empty()) {
			all_lead_points.push_back(cv::Point2f(x, y));
			all_lead_track_ids.push_back(selected_track_id);
		}

		const double travel_px =
			std::abs(static_cast<double>(x) - static_cast<double>(x_center));
		const double flight_time_s = travel_px / bullet_speed;
		const double lead_time_dynamic_s =
			std::clamp(flight_time_s + latency_comp_s, 0.0, 0.5);

		cv::Point2f lead_point(x, y);
		auto lead_it = lead_point_by_track.find(selected_track_id);
		if (lead_it != lead_point_by_track.end()) {
			lead_point = lead_it->second;
		}
		const float lead_x = lead_point.x;
		maybeSaveDebugImages(frame, crop, selected, &lead_point, &all_lead_points,
									&all_lead_track_ids);

		const float norm = (lead_x - x_center) / x_center;
		const float yaw_body =
			static_cast<float>(yaw_sign_ * yaw_fov_deg_ * 0.5 * norm);
		float yaw = yaw_body + static_cast<float>(yaw_offset_deg_);
		while (yaw > 180.0f) yaw -= 360.0f;
		while (yaw < -180.0f) yaw += 360.0f;

		rclcpp::Clock steady_clock(RCL_STEADY_TIME);
		const rclcpp::Time now_steady = steady_clock.now();
		const int yaw_interval = std::max(1, yaw_send_interval_ms_);
		const bool yaw_due =
			(now_steady - last_yaw_send_time_).nanoseconds() >=
			static_cast<int64_t>(yaw_interval) * 1000000LL;
		if (yaw_due) {
			if (!sender_.isOpen()) {
				sender_.open();
			}
			if (sender_.isOpen()) {
				sender_.sendYawDegrees(yaw);
				last_yaw_send_time_ = now_steady;
			} else {
				RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
					"Fire blocked: serial not open (%s)", serial_device_.c_str());
			}
		}

		if (!auto_fire_) {
			stop_firing_ = true;
			RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
				"Fire blocked: auto_fire=false");
		}

		if ((own_color == "red" || own_color == "blue") &&
			selected.label == own_color) {
			stop_firing_ = true;
			RCLCPP_INFO_THROTTLE(
				get_logger(), *get_clock(), 500,
				"Fire blocked: target color equals own color (%s)",
				own_color.c_str());
		} else if (auto_fire_) {
			stop_firing_ = false;
		}

		const double lock_d =
			last_selected_valid_
				? std::hypot(static_cast<double>(x - last_selected_center_.x),
							 static_cast<double>(y - last_selected_center_.y))
				: std::numeric_limits<double>::infinity();
		if (last_selected_valid_ && lock_d < 40.0) {
			target_lock_frames_ = std::min(target_lock_frames_ + 1, 20);
		} else {
			target_lock_frames_ = 1;
		}
		last_selected_center_ = cv::Point2f(x, y);
		last_selected_valid_ = true;
		if (last_selected_track_id_ != selected_track_id) {
			last_track_switch_time_ = now;
		}
		last_selected_track_id_ = selected_track_id;

		RCLCPP_INFO_THROTTLE(
			get_logger(), *get_clock(), 500,
			"Own=%s Target=%s center=(%.1f, %.1f) yaw=%.1f lead_t=%.3f",
			own_color.c_str(), selected.label.c_str(), lead_x, y_lb, yaw,
			lead_time_dynamic_s);
	}

	void onFireTimer() {
		if (!auto_fire_ || stop_firing_) {
			return;
		}

		if (!sender_.isOpen()) {
			sender_.open();
		}
		if (!sender_.isOpen()) {
			RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
				"Fire blocked: serial not open (%s)", serial_device_.c_str());
			return;
		}

		if (sender_.sendFire()) {
			if (last_selected_track_id_ >= 0) {
				TrackState *track = findTrackById(last_selected_track_id_);
				if (track) {
					track->last_fire = this->now();
				}
			}
			RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 200,
				"Fire TX: timer period=%dms", std::max(1, fire_cooldown_ms_));
		} else {
			RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
				"Fire blocked: sendFire() failed");
		}
	}

 private:
	struct TrackState {
		int id = -1;
		cv::Point2f center{0.0f, 0.0f};
		cv::Point2f velocity{0.0f, 0.0f};
		cv::Rect bbox;
		rclcpp::Time last_seen{0, 0, RCL_ROS_TIME};
		rclcpp::Time last_fire{0, 0, RCL_ROS_TIME};
		std::string label;
	};

	TrackState *findTrackById(int id) {
		for (auto &t : tracks_) {
			if (t.id == id) {
				return &t;
			}
		}
		return nullptr;
	}

	void pruneTracks(const rclcpp::Time &now) {
		const int64_t ttl_ns = 1200LL * 1000000LL;
		tracks_.erase(
			std::remove_if(tracks_.begin(), tracks_.end(),
				[&](const TrackState &t) {
					if (t.last_seen.nanoseconds() <= 0) {
						return true;
					}
					return (now - t.last_seen).nanoseconds() > ttl_ns;
				}),
			tracks_.end());

		for (auto it = track_kfs_.begin(); it != track_kfs_.end();) {
			if (findTrackById(it->first) == nullptr) {
				track_kf_last_meas_time_.erase(it->first);
				it = track_kfs_.erase(it);
			} else {
				++it;
			}
		}
	}

	int matchOrCreateTrack(const cv::Point2f &center, const rclcpp::Time &now) {
		double best_d = std::numeric_limits<double>::infinity();
		int best_idx = -1;
		for (size_t i = 0; i < tracks_.size(); ++i) {
			double dt = 0.0;
			if (tracks_[i].last_seen.nanoseconds() > 0) {
				dt = (now - tracks_[i].last_seen).seconds();
			}
			dt = std::clamp(dt, 0.0, 0.35);
			const cv::Point2f pred(
				tracks_[i].center.x + tracks_[i].velocity.x * static_cast<float>(dt),
				tracks_[i].center.y + tracks_[i].velocity.y * static_cast<float>(dt));
			const double d = std::hypot(
				static_cast<double>(center.x - pred.x),
				static_cast<double>(center.y - pred.y));
			if (d < best_d) {
				best_d = d;
				best_idx = static_cast<int>(i);
			}
		}

		if (best_idx >= 0 && best_d <= std::max(5.0, track_match_px_)) {
			double dt = 0.0;
			if (tracks_[best_idx].last_seen.nanoseconds() > 0) {
				dt = (now - tracks_[best_idx].last_seen).seconds();
			}
			if (dt > 1e-3) {
				tracks_[best_idx].velocity = cv::Point2f(
					(center.x - tracks_[best_idx].center.x) / static_cast<float>(dt),
					(center.y - tracks_[best_idx].center.y) / static_cast<float>(dt));
			}
			tracks_[best_idx].center = center;
			tracks_[best_idx].last_seen = now;
			return tracks_[best_idx].id;
		}

		TrackState t;
		t.id = next_track_id_++;
		t.center = center;
		t.last_seen = now;
		tracks_.push_back(t);
		return t.id;
	}

	cv::Point2f compensateLeadByY(const cv::Point2f &meas_point,
							 const cv::Point2f &lead_point,
							 int frame_cols,
							 int frame_rows) const {
		if (frame_rows <= 1 || frame_cols <= 1) {
			return lead_point;
		}

		const double y01 = std::clamp(
			static_cast<double>(meas_point.y) /
				static_cast<double>(frame_rows - 1),
			0.0, 1.0);
		const double start = std::clamp(lead_y_comp_start_ratio_, 0.0, 1.0);
		if (y01 <= start || start >= 1.0) {
			return lead_point;
		}

		const double blend = std::clamp((y01 - start) / (1.0 - start), 0.0, 1.0);
		const double bottom_scale = std::clamp(lead_y_comp_scale_bottom_, 0.0, 1.0);
		const double y_scale = 1.0 - (1.0 - bottom_scale) * blend;

		const double x_center = (static_cast<double>(frame_cols) - 1.0) * 0.5;
		const double center_norm = std::abs(
			static_cast<double>(meas_point.x) - x_center) /
			std::max(1.0, x_center);
		const double center_window = std::clamp(lead_center_window_ratio_, 0.01, 1.0);
		const double center_blend = std::clamp(1.0 - center_norm / center_window, 0.0, 1.0);
		const double center_min = std::clamp(lead_center_scale_min_, 0.0, 1.0);
		const double center_scale = 1.0 - (1.0 - center_min) * center_blend;

		const double dx = static_cast<double>(lead_point.x - meas_point.x);
		double compensated_dx = dx * y_scale * center_scale;

		const double fixed_pull_raw =
			std::max(0.0, lead_y_comp_fixed_px_bottom_) * blend;
		const double fixed_pull_cap = std::abs(compensated_dx) * 0.45;
		const double fixed_pull = std::min(fixed_pull_raw, fixed_pull_cap);
		if (compensated_dx > 0.0) {
			compensated_dx = std::max(0.0, compensated_dx - fixed_pull);
		} else if (compensated_dx < 0.0) {
			compensated_dx = std::min(0.0, compensated_dx + fixed_pull);
		}

		cv::Point2f out = lead_point;
		out.x = static_cast<float>(meas_point.x + compensated_dx);
		return out;
	}

	std::string serial_device_;
	SerialSender sender_;
	rclcpp::TimerBase::SharedPtr fire_timer_;
	double bullet_speed_px_s_;
	double enemy_speed_max_px_s_;
	double track_match_px_;
	int target_commit_ms_;
	int hold_lost_grace_ms_;
	double switch_score_margin_;
	int switch_min_interval_ms_;
	double yaw_move_penalty_w_;
	double center_bias_w_;
	double yaw_fov_deg_;
	double yaw_sign_;
	double yaw_offset_deg_;
	int yaw_send_interval_ms_;
	int fire_x_tol_px_;
	int fire_cooldown_ms_;
	int edge_ignore_px_;
	double lead_time_s_;
	double lead_y_comp_start_ratio_;
	double lead_y_comp_scale_bottom_;
	double lead_y_comp_fixed_px_bottom_;
	double lead_center_window_ratio_;
	double lead_center_scale_min_;
	double kf_sigma_a_;
	double kf_sigma_z_;
	int target_reset_ms_;
	bool auto_fire_;
	rclcpp::Time last_fire_time_;
	rclcpp::Time last_target_lost_time_;
	rclcpp::Time last_yaw_send_time_;
	cv::Point2f last_selected_center_{0.0f, 0.0f};
	bool last_selected_valid_ = false;
	int last_selected_track_id_ = -1;
	rclcpp::Time last_track_switch_time_{0, 0, RCL_ROS_TIME};
	int target_lock_frames_ = 0;
	std::vector<TrackState> tracks_;
	std::unordered_map<int, KalmanFilterCV2D> track_kfs_;
	std::unordered_map<int, rclcpp::Time> track_kf_last_meas_time_;
	int next_track_id_ = 1;
	bool stop_firing_;
};

}  // namespace cilent

int main(int argc, char **argv) {
	rclcpp::init(argc, argv);
	auto node = std::make_shared<cilent::AutoAimNode>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
