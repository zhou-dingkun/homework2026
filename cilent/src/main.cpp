#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <memory>
#include <string>
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
														 "/dev/pts/8")),
			sender_(serial_device_),
			bullet_speed_px_s_(declare_parameter<double>("bullet_speed_px_s", 600.0)),
			enemy_speed_max_px_s_(
				declare_parameter<double>("enemy_speed_max_px_s", 300.0)),
			gray_s_max_(declare_parameter<int>("gray_s_max", 55)),
			gray_v_min_(declare_parameter<int>("gray_v_min", 60)),
			hit_gray_ratio_thresh_(
				declare_parameter<double>("hit_gray_ratio_thresh", 0.45)),
			hit_confirm_window_ms_(
				declare_parameter<int>("hit_confirm_window_ms", 220)),
			gray_cooldown_ms_(declare_parameter<int>("gray_cooldown_ms", 260)),
			track_match_px_(declare_parameter<double>("track_match_px", 90.0)),
			target_commit_ms_(declare_parameter<int>("target_commit_ms", 220)),
			hold_lost_grace_ms_(declare_parameter<int>("hold_lost_grace_ms", 260)),
			switch_score_margin_(declare_parameter<double>("switch_score_margin", 0.9)),
			switch_min_interval_ms_(
				declare_parameter<int>("switch_min_interval_ms", 180)),
			yaw_move_penalty_w_(declare_parameter<double>("yaw_move_penalty_w", 1.1)),
			center_bias_w_(declare_parameter<double>("center_bias_w", 0.25)),
			yaw_fov_deg_(declare_parameter<double>("yaw_fov_deg", 90.0)),
			yaw_sign_(declare_parameter<double>("yaw_sign", -1.0)),
			yaw_offset_deg_(declare_parameter<double>("yaw_offset_deg", 90.0)),
			yaw_send_interval_ms_(declare_parameter<int>("yaw_send_interval_ms", 120)),
			fire_x_tol_px_(declare_parameter<int>("fire_x_tol_px", 60)),
			fire_cooldown_ms_(declare_parameter<int>("fire_cooldown_ms", 50)),
			edge_ignore_px_(declare_parameter<int>("edge_ignore_px", 20)),
			lead_time_s_(declare_parameter<double>("lead_time_s", 0.03)),
			kf_sigma_a_(declare_parameter<double>("kf_sigma_a", 120.0)),
			kf_sigma_z_(declare_parameter<double>("kf_sigma_z", 6.0)),
			target_reset_ms_(declare_parameter<int>("target_reset_ms", 500)),
			auto_fire_(declare_parameter<bool>("auto_fire", true)),
			last_fire_time_(0, 0, RCL_ROS_TIME),
			last_target_lost_time_(0, 0, RCL_ROS_TIME),
			last_meas_time_(0, 0, RCL_ROS_TIME),
			last_yaw_send_time_(0, 0, RCL_STEADY_TIME),
			stop_firing_(false),
			kf_(0.02, kf_sigma_a_, kf_sigma_z_),
			kf_initialized_(false) {
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
		if (!detection.boxes.empty()) {
			const float frame_cx = (static_cast<float>(frame.cols) - 1.0f) * 0.5f;
			const float frame_cy = (static_cast<float>(frame.rows) - 1.0f) * 0.5f;
			const double roi_area =
				static_cast<double>(std::max(1, crop.width * crop.height));

			double best_score = -std::numeric_limits<double>::infinity();
			bool best_in_gray = false;
			cv::Rect best_box = detection.bbox;
			cv::Point2f best_center = detection.center;

			double current_score = -std::numeric_limits<double>::infinity();
			bool current_visible = false;
			bool current_in_gray = false;
			cv::Rect current_box = detection.bbox;
			cv::Point2f current_center = detection.center;

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

				const double gray_ratio = computeGrayRatio(frame, track->bbox);
				const bool gray_now = gray_ratio >= hit_gray_ratio_thresh_;
				const bool gray_rise =
					gray_now && track->last_gray_ratio < hit_gray_ratio_thresh_;
				const bool fire_recent =
					track->last_fire.nanoseconds() > 0 &&
					(now - track->last_fire).nanoseconds() <=
						static_cast<int64_t>(std::max(1, hit_confirm_window_ms_)) *
							1000000LL;
				if (gray_rise && fire_recent) {
					track->last_hit = now;
					track->hit_count += 1;
					RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 300,
						"Hit confirmed: track=%d hit_count=%d gray_ratio=%.2f",
						track->id, track->hit_count, gray_ratio);
				}
				track->last_gray_ratio = gray_ratio;

				const bool in_gray_cooldown =
					track->last_hit.nanoseconds() > 0 &&
					(now - track->last_hit).nanoseconds() <=
						static_cast<int64_t>(std::max(1, gray_cooldown_ms_)) * 1000000LL;

				// 1) 贴边目标优先：降低“丢失敌方目标 -1 分”风险
				const float dx_edge =
					std::min(cx, static_cast<float>(frame.cols - 1) - cx);
				const float dy_edge =
					std::min(cy, static_cast<float>(frame.rows - 1) - cy);
				const float min_edge = std::max(0.0f, std::min(dx_edge, dy_edge));
				const double t_edge =
					static_cast<double>(min_edge) /
					std::max(1.0, enemy_speed_max_px_s_);
				const double edge_urgency = 1.0 / (0.08 + t_edge);

				// 2) 命中概率：中心更稳定，且考虑飞行时间内目标可能位移
				const double center_dist_norm =
					std::abs(static_cast<double>(cx - frame_cx)) /
					std::max(1.0, static_cast<double>(frame_cx));
				const double center_score =
					std::clamp(1.0 - center_dist_norm, 0.0, 1.0);

				const double yaw_move_penalty =
					last_selected_valid_
						? std::abs(static_cast<double>(cx - last_selected_center_.x)) /
							  std::max(1.0, static_cast<double>(frame.cols))
						: 0.0;

				const double flight_t =
					std::abs(static_cast<double>(frame_cy - cy)) /
					std::max(1.0, bullet_speed_px_s_);
				const double drift_px = enemy_speed_max_px_s_ * flight_t;
				const double hit_prob_score = 1.0 / (1.0 + drift_px / 35.0);

				// 3) 保持火力连续：尽量先打完当前目标（完整击毁收益更高）
				double finish_bonus = 0.0;
				if (!in_gray_cooldown && last_selected_track_id_ == track_id) {
					finish_bonus = 1.2 + 0.2 * std::min(10, target_lock_frames_);
				} else if (!in_gray_cooldown && last_selected_valid_) {
					const double d = std::hypot(
						static_cast<double>(cx - last_selected_center_.x),
						static_cast<double>(cy - last_selected_center_.y));
					const double scale =
						std::max(12.0, static_cast<double>(std::max(box.width, box.height)));
					if (d < 1.6 * scale) {
						finish_bonus = 0.8 + 0.2 * std::min(8, target_lock_frames_);
					}
				}

				const double gray_penalty = 3.4 * (in_gray_cooldown ? 1.0 : 0.0);
				const double gray_soft_penalty = 1.8 * gray_ratio;
				const double switch_bonus =
					(last_selected_track_id_ >= 0 && last_selected_track_id_ != track_id &&
					 !in_gray_cooldown)
						? 0.9
						: 0.0;

				double commit_bonus = 0.0;
				if (last_selected_track_id_ == track_id &&
					last_track_switch_time_.nanoseconds() > 0 &&
					(now - last_track_switch_time_).nanoseconds() <=
						static_cast<int64_t>(std::max(1, target_commit_ms_)) *
							1000000LL) {
					commit_bonus = 1.2;
				}

				const double area_score = static_cast<double>(box.area()) / roi_area;
				const double total_score =
					2.5 * edge_urgency + center_bias_w_ * center_score +
					1.6 * hit_prob_score + 2.0 * finish_bonus + 0.8 * area_score +
					switch_bonus + commit_bonus - gray_penalty - gray_soft_penalty -
					yaw_move_penalty_w_ * yaw_move_penalty;

				if (total_score > best_score) {
					best_score = total_score;
					best_in_gray = in_gray_cooldown;
					best_box = box;
					selected_track_id = track_id;
					best_center = cv::Point2f(
						static_cast<float>(box.x + box.width * 0.5f),
						static_cast<float>(box.y + box.height * 0.5f));
				}

				if (track_id == last_selected_track_id_) {
					current_visible = true;
					current_in_gray = in_gray_cooldown;
					current_score = total_score;
					current_box = box;
					current_center = cv::Point2f(
						static_cast<float>(box.x + box.width * 0.5f),
						static_cast<float>(box.y + box.height * 0.5f));
				}
			}

			if (current_visible && last_selected_track_id_ >= 0 &&
				selected_track_id != last_selected_track_id_) {
				const bool recent_switch =
					last_track_switch_time_.nanoseconds() > 0 &&
					(now - last_track_switch_time_).nanoseconds() <=
						static_cast<int64_t>(std::max(1, switch_min_interval_ms_)) *
							1000000LL;
				const double lock_margin =
					switch_score_margin_ + 0.15 * std::min(12, target_lock_frames_);
				const bool strong_better = best_score > current_score + lock_margin;
				const bool keep_current =
					recent_switch || (!strong_better && !current_in_gray) ||
					(best_in_gray && !current_in_gray);
				if (keep_current) {
					selected_track_id = last_selected_track_id_;
					best_box = current_box;
					best_center = current_center;
				}
			}

			selected.bbox = best_box;
			selected.center = best_center;
			selected.area = static_cast<double>(best_box.area());
		}

		const float x = selected.center.x + static_cast<float>(crop.x);
		const float y = selected.center.y + static_cast<float>(crop.y);
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

		if (!kf_initialized_) {
			kf_.reset(x, y, 0.0, 0.0);
			kf_initialized_ = true;
			last_meas_time_ = now;
		} else {
			const double dt = (now - last_meas_time_).seconds();
			if (dt > 0.0) {
				kf_.setDeltaT(dt);
			}
			kf_.setProcessNoise(kf_sigma_a_);
			kf_.setMeasurementNoise(kf_sigma_z_);
			kf_.predict();
			kf_.update(x, y);
			last_meas_time_ = now;
		}

		const Eigen::Vector2d lead = kf_.predictAhead(lead_time_s_);
		const float lead_x = static_cast<float>(lead.x());
		const float lead_y = static_cast<float>(lead.y());
		const cv::Point2f lead_point(lead_x, lead_y);
		maybeSaveDebugImages(frame, crop, selected, &lead_point);

		const float x_center = (static_cast<float>(frame.cols) - 1.0f) * 0.5f;
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
			"Own=%s Target=%s center=(%.1f, %.1f) yaw=%.1f",
			own_color.c_str(), selected.label.c_str(), lead_x, y_lb, yaw);
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
		rclcpp::Time last_hit{0, 0, RCL_ROS_TIME};
		double last_gray_ratio = 0.0;
		int hit_count = 0;
		std::string label;
	};

	double computeGrayRatio(const cv::Mat &frame, const cv::Rect &rect) const {
		cv::Rect r = rect & cv::Rect(0, 0, frame.cols, frame.rows);
		if (r.area() <= 0) {
			return 0.0;
		}

		cv::Mat roi = frame(r);
		if (roi.empty()) {
			return 0.0;
		}

		cv::Mat hsv;
		cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
		cv::Mat low_s, gray_mask;
		cv::inRange(hsv, cv::Scalar(0, 0, std::max(0, gray_v_min_)),
					cv::Scalar(180, std::max(0, gray_s_max_), 255), low_s);
		gray_mask = low_s;
		const int total = std::max(1, r.area());
		const int gray = cv::countNonZero(gray_mask);
		return static_cast<double>(gray) / static_cast<double>(total);
	}

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

	std::string serial_device_;
	SerialSender sender_;
	rclcpp::TimerBase::SharedPtr fire_timer_;
	double bullet_speed_px_s_;
	double enemy_speed_max_px_s_;
	int gray_s_max_;
	int gray_v_min_;
	double hit_gray_ratio_thresh_;
	int hit_confirm_window_ms_;
	int gray_cooldown_ms_;
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
	double kf_sigma_a_;
	double kf_sigma_z_;
	int target_reset_ms_;
	bool auto_fire_;
	rclcpp::Time last_fire_time_;
	rclcpp::Time last_target_lost_time_;
	rclcpp::Time last_meas_time_;
	rclcpp::Time last_yaw_send_time_;
	cv::Point2f last_selected_center_{0.0f, 0.0f};
	bool last_selected_valid_ = false;
	int last_selected_track_id_ = -1;
	rclcpp::Time last_track_switch_time_{0, 0, RCL_ROS_TIME};
	int target_lock_frames_ = 0;
	std::vector<TrackState> tracks_;
	int next_track_id_ = 1;
	bool stop_firing_;
	KalmanFilterCV2D kf_;
	bool kf_initialized_;
};

}  // namespace cilent

int main(int argc, char **argv) {
	rclcpp::init(argc, argv);
	auto node = std::make_shared<cilent::AutoAimNode>();
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
