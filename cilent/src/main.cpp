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
			lead_time_s_(declare_parameter<double>("lead_time_s", 0.0)),
			kf_sigma_a_(declare_parameter<double>("kf_sigma_a", 120.0)),
			kf_sigma_z_(declare_parameter<double>("kf_sigma_z", 0.1)),
			target_reset_ms_(declare_parameter<int>("target_reset_ms", 300)),
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
		const float x_center = (static_cast<float>(frame.cols) - 1.0f) * 0.5f;
		const double bullet_speed = std::max(1.0, bullet_speed_px_s_);
		const double latency_comp_s = std::max(0.0, lead_time_s_);
		std::vector<cv::Point2f> all_lead_points;
		all_lead_points.reserve(detection.boxes.size());
		if (!detection.boxes.empty()) {
			int earliest_any_track_id = std::numeric_limits<int>::max();
			bool has_any_target = false;
			cv::Rect earliest_any_box;
			cv::Point2f earliest_any_center;

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

				const double travel_px =
					std::abs(static_cast<double>(track->center.x) -
							 static_cast<double>(x_center));
				const double flight_time_s = travel_px / bullet_speed;
				const double lead_time_dynamic_s =
					std::clamp(flight_time_s + latency_comp_s, 0.0, 0.5);
				const cv::Point2f lead_point(
					track->center.x +
						track->velocity.x * static_cast<float>(lead_time_dynamic_s),
					track->center.y +
						track->velocity.y * static_cast<float>(lead_time_dynamic_s));
				all_lead_points.push_back(lead_point);

				const cv::Point2f center_roi(
					static_cast<float>(box.x + box.width * 0.5f),
					static_cast<float>(box.y + box.height * 0.5f));

				if (track_id < earliest_any_track_id) {
					earliest_any_track_id = track_id;
					earliest_any_box = box;
					earliest_any_center = center_roi;
					has_any_target = true;
				}
			}

			if (has_any_target) {
				selected_track_id = earliest_any_track_id;
				selected.bbox = earliest_any_box;
				selected.center = earliest_any_center;
				selected.area = static_cast<double>(earliest_any_box.area());
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
			kf_.reset(x, y, 0.0, 0.0);
			kf_initialized_ = true;
			last_meas_time_ = now;
			RCLCPP_INFO_THROTTLE(
				get_logger(), *get_clock(), 300,
				"Track switched (%d->%d), KF reset at (%.1f, %.1f)",
				last_selected_track_id_, selected_track_id, x, y_lb);
		}

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

		if (all_lead_points.empty()) {
			all_lead_points.push_back(cv::Point2f(x, y));
		}

		const double travel_px =
			std::abs(static_cast<double>(x) - static_cast<double>(x_center));
		const double flight_time_s = travel_px / bullet_speed;
		const double lead_time_dynamic_s =
			std::clamp(flight_time_s + latency_comp_s, 0.0, 0.5);

		const Eigen::Vector2d lead = kf_.predictAhead(lead_time_dynamic_s);
		const float lead_x = static_cast<float>(lead.x());
		const float lead_y = static_cast<float>(lead.y());
		const cv::Point2f lead_point(lead_x, lead_y);
		maybeSaveDebugImages(frame, crop, selected, &lead_point, &all_lead_points);

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
