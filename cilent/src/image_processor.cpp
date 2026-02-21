#include "cilent/image_processor.h"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace cilent {

ImageProcessorNode::ImageProcessorNode(const rclcpp::NodeOptions &options)
  : rclcpp::Node("image_processor", options) {
  expected_width_ = declare_parameter<int>("expected_width", 1152);
  expected_height_ = declare_parameter<int>("expected_height", 648);
  crop_width_ = declare_parameter<int>("crop_width", 400);
  crop_height_ = declare_parameter<int>("crop_height", 200);
  center_window_ = declare_parameter<int>("center_window", 7);
  color_delta_ = declare_parameter<int>("color_delta", 30);

  sub_ = create_subscription<sensor_msgs::msg::Image>(
    "image_raw", rclcpp::SensorDataQoS(),
    std::bind(&ImageProcessorNode::onImage, this, std::placeholders::_1));
}

void ImageProcessorNode::onImage(
  const sensor_msgs::msg::Image::SharedPtr msg) {
    if (!msg) {
      return;
    }

    if (msg->width != static_cast<uint32_t>(expected_width_) ||
        msg->height != static_cast<uint32_t>(expected_height_)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "Unexpected resolution: %ux%u, expected %dx%d",
                           msg->width, msg->height, expected_width_,
                           expected_height_);
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

    cv::Rect crop = computeCropRect(frame.cols, frame.rows);
    if (crop.width <= 0 || crop.height <= 0) {
      return;
    }

    cv::Mat roi = frame(crop);
    const cv::Vec3d mean_bgr = meanCenterColor(roi);
    const std::string color = classifyColor(mean_bgr);

    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
                         "Center color: %s (B=%.1f G=%.1f R=%.1f)",
                         color.c_str(), mean_bgr[0], mean_bgr[1], mean_bgr[2]);
  }

}

cv::Rect ImageProcessorNode::computeCropRect(int width, int height) const {
  int crop_w = std::clamp(crop_width_, 1, width);
  int crop_h = std::clamp(crop_height_, 1, height);

  int x = std::max(0, (width - crop_w) / 2);
  int y = std::max(0, height - crop_h);

  if (x + crop_w > width) {
    crop_w = width - x;
  }
  if (y + crop_h > height) {
    crop_h = height - y;
  }

  return cv::Rect(x, y, crop_w, crop_h);
}

cv::Vec3d ImageProcessorNode::meanCenterColor(const cv::Mat &roi) const {
  const int w = roi.cols;
  const int h = roi.rows;
  const int half = std::max(1, center_window_ / 2);

  const int cx = w / 2;
  const int cy = h / 2;
  const int x0 = std::clamp(cx - half, 0, w - 1);
  const int y0 = std::clamp(cy - half, 0, h - 1);
  const int x1 = std::clamp(cx + half, 0, w - 1);
  const int y1 = std::clamp(cy + half, 0, h - 1);

  cv::Rect center_rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
  cv::Mat center = roi(center_rect);
  return cv::mean(center);
}

std::string ImageProcessorNode::classifyColor(const cv::Vec3d &bgr) const {
  const double b = bgr[0];
  const double g = bgr[1];
  const double r = bgr[2];

  const double max_bg = std::max(b, g);
  if (r > max_bg + color_delta_) {
    return "red";
  }
  const double max_br = std::max(b, r);
  if (g > max_br + color_delta_) {
    return "green";
  }
  const double max_gr = std::max(g, r);
  if (b > max_gr + color_delta_) {
    return "blue";
  }
  return "unknown";
}

}  // namespace cilent
