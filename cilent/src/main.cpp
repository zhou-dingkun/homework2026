#include <cmath>
#include <memory>
#include <string>

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
			sender_("/dev/pts/8"),
			yaw_fov_deg_(declare_parameter<double>("yaw_fov_deg", 90.0)),
			fire_x_tol_px_(declare_parameter<int>("fire_x_tol_px", 10)),
			fire_cooldown_ms_(declare_parameter<int>("fire_cooldown_ms", 250)),
			edge_ignore_px_(declare_parameter<int>("edge_ignore_px", 100)),
			lead_time_s_(declare_parameter<double>("lead_time_s", 0.12)),
			kf_sigma_a_(declare_parameter<double>("kf_sigma_a", 120.0)),
			kf_sigma_z_(declare_parameter<double>("kf_sigma_z", 6.0)),
			target_reset_ms_(declare_parameter<int>("target_reset_ms", 500)),
			auto_fire_(declare_parameter<bool>("auto_fire", true)),
			last_fire_time_(0, 0, RCL_ROS_TIME),
			last_target_lost_time_(0, 0, RCL_ROS_TIME),
			last_meas_time_(0, 0, RCL_ROS_TIME),
			stop_firing_(false),
			kf_(0.02, kf_sigma_a_, kf_sigma_z_),
			kf_initialized_(false) {
		if (!sender_.open()) {
			RCLCPP_WARN(get_logger(), "Failed to open serial device %s",
									"/dev/pts/8");
		}
	}

 protected:
	void onImage(const sensor_msgs::msg::Image::SharedPtr msg) override {
		if (!msg) {
			return;
		}

		if (msg->width != static_cast<uint32_t>(expected_width_) ||
				msg->height != static_cast<uint32_t>(expected_height_)) {
			RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
													 "Unexpected resolution: %ux%u, expected %dx%d",
													 msg->width, msg->height, expected_width_,
													 expected_height_);
			return;
		}

		cv_bridge::CvImageConstPtr cv_ptr;
		try {
			cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
		} catch (const cv_bridge::Exception &ex) {
			RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", ex.what());
			return;
		}

		const cv::Mat &frame = cv_ptr->image;
		if (frame.empty()) {
			return;
		}

		const cv::Rect crop = computeCropRect(frame.cols, frame.rows);
		if (crop.width <= 0 || crop.height <= 0) {
			return;
		}

		const cv::Mat roi = frame(crop);
		const cv::Vec3d mean_bgr = meanCenterColor(roi);
		const std::string own_color = classifyColor(mean_bgr);
		const DetectionResult detection = detectTarget(roi, own_color);

		if (!detection.found) {
			last_target_lost_time_ = this->now();
			stop_firing_ = true;
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

		const float x = detection.center.x + static_cast<float>(crop.x);
		const float y = detection.center.y + static_cast<float>(crop.y);
		const int edge = std::max(0, edge_ignore_px_);
		if (x < static_cast<float>(edge) ||
				x > static_cast<float>(frame.cols - 1 - edge)) {
			return;
		}
		const float y_lb = static_cast<float>(frame.rows - 1) - y;

		const rclcpp::Time now = this->now();
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

		const float x_center = (static_cast<float>(frame.cols) - 1.0f) * 0.5f;
		const float norm = (lead_x - x_center) / x_center;
		const float yaw = static_cast<float>(yaw_fov_deg_ * 0.5 * norm);

		if (!sender_.isOpen()) {
			sender_.open();
		}
		if (sender_.isOpen()) {
			sender_.sendYawDegrees(yaw);
		}

		bool allow_fire = auto_fire_ && !stop_firing_;
		if (own_color == "red" || own_color == "blue") {
			if (detection.label == own_color) {
				allow_fire = false;
			}
		} else {
			allow_fire = false;
		}

		if (allow_fire && sender_.isOpen()) {
			const float dx = std::abs(lead_x - x_center);
			const bool cooldown_ok =
				(now - last_fire_time_).nanoseconds() >=
				static_cast<int64_t>(fire_cooldown_ms_) * 1000000LL;
			if (dx <= static_cast<float>(fire_x_tol_px_) && cooldown_ok) {
				sender_.sendFire();
				last_fire_time_ = now;
			}
		}

		RCLCPP_INFO_THROTTLE(
			get_logger(), *get_clock(), 500,
			  "Own=%s Target=%s center=(%.1f, %.1f) yaw=%.1f", 
			  own_color.c_str(), detection.label.c_str(), lead_x, y_lb, yaw);
	}

 private:
	SerialSender sender_;
	double yaw_fov_deg_;
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
