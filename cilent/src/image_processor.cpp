#include "cilent/image_processor.h"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
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

  hsv_s_min_ = declare_parameter<int>("hsv_s_min", 60);
  hsv_v_min_ = declare_parameter<int>("hsv_v_min", 60);
  red_h_low1_ = declare_parameter<int>("red_h_low1", 0);
  red_h_high1_ = declare_parameter<int>("red_h_high1", 10);
  red_h_low2_ = declare_parameter<int>("red_h_low2", 170);
  red_h_high2_ = declare_parameter<int>("red_h_high2", 180);
  blue_h_low_ = declare_parameter<int>("blue_h_low", 95);
  blue_h_high_ = declare_parameter<int>("blue_h_high", 135);
  morph_kernel_ = declare_parameter<int>("morph_kernel", 3);
  marker_min_area_ = declare_parameter<int>("marker_min_area", 40);
  marker_max_area_ = declare_parameter<int>("marker_max_area", 400);
  pair_y_tol_ = declare_parameter<int>("pair_y_tol", 10);
  pair_min_dx_ = declare_parameter<int>("pair_min_dx", 50);
  bbox_pad_ = declare_parameter<int>("bbox_pad", 6);
  debug_save_path_ =
    declare_parameter<std::string>("debug_save_path",
                                   "center_color_overlay.png");
  debug_save_interval_ms_ =
    declare_parameter<int>("debug_save_interval_ms", 500);
  debug_save_overlay_ = declare_parameter<bool>("debug_save_overlay", true);
  last_debug_save_time_ = rclcpp::Time(0, 0, RCL_STEADY_TIME);
  debug_save_frames_ = declare_parameter<bool>("debug_save_frames", true);
  debug_frames_dir_ =
    declare_parameter<std::string>("debug_frames_dir", "frames");
  debug_frames_interval_ms_ =
    declare_parameter<int>("debug_frames_interval_ms", 100);
  debug_frame_index_ = 0;
  last_debug_frame_time_ = rclcpp::Time(0, 0, RCL_STEADY_TIME);

  std::filesystem::path frames_path(debug_frames_dir_);
  if (frames_path.is_relative()) {
    frames_path = std::filesystem::current_path() / frames_path;
  }
  debug_frames_dir_resolved_ = frames_path.string();

  RCLCPP_INFO(get_logger(),
              "Debug frames: enabled=%s dir=%s interval_ms=%d",
              debug_save_frames_ ? "true" : "false",
              debug_frames_dir_resolved_.c_str(),
              debug_frames_interval_ms_);

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

  cv::Rect crop = computeCropRect(frame.cols, frame.rows);
  if (crop.width <= 0 || crop.height <= 0) {
    return;
  }

  cv::Mat roi = frame(crop);
  const cv::Vec3d mean_bgr = meanCenterColor(frame);
  const std::string color = classifyColor(mean_bgr);
  DetectionResult detection = detectTarget(roi, color);

  if (debug_save_frames_) {
    rclcpp::Clock steady_clock(RCL_STEADY_TIME);
    const rclcpp::Time now = steady_clock.now();
    const bool due =
      debug_frames_interval_ms_ <= 0 ||
      (now - last_debug_frame_time_).nanoseconds() >=
        static_cast<int64_t>(debug_frames_interval_ms_) * 1000000LL;
    if (due) {
      std::error_code ec;
      std::filesystem::create_directories(debug_frames_dir_resolved_, ec);
      if (ec) {
        RCLCPP_WARN(get_logger(), "Failed to create debug frame dir %s: %s",
                    debug_frames_dir_resolved_.c_str(),
                    ec.message().c_str());
      } else {
        std::ostringstream name;
        name << debug_frames_dir_resolved_ << "/frame_"
             << std::setfill('0')
             << std::setw(6) << debug_frame_index_++ << ".png";
        if (!cv::imwrite(name.str(), frame)) {
          RCLCPP_WARN(get_logger(), "Failed to write debug frame to %s",
                      name.str().c_str());
        } else {
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                               "Saved debug frame to %s",
                               name.str().c_str());
        }
      }
      last_debug_frame_time_ = now;
    }
  }

  if (debug_save_overlay_) {
    rclcpp::Clock steady_clock(RCL_STEADY_TIME);
    const rclcpp::Time now = steady_clock.now();
    const bool due =
      (now - last_debug_save_time_).nanoseconds() >=
      static_cast<int64_t>(debug_save_interval_ms_) * 1000000LL;
    if (due) {
      cv::Mat debug_frame = frame.clone();
      const cv::Rect sample_rect = sampleRect(frame);
      if (sample_rect.area() > 0) {
        cv::rectangle(debug_frame, sample_rect, cv::Scalar(0, 255, 0), 2);
      }
      if (!cv::imwrite(debug_save_path_, debug_frame)) {
        RCLCPP_WARN(get_logger(), "Failed to write debug overlay to %s",
                    debug_save_path_.c_str());
      } else {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Saved debug overlay to %s",
                             debug_save_path_.c_str());
      }
      last_debug_save_time_ = now;
    }
  }

  if (detection.found) {
    const float x = detection.center.x + static_cast<float>(crop.x);
    const float y = detection.center.y + static_cast<float>(crop.y);
    const float y_lb = static_cast<float>(frame.rows - 1) - y;
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 500,
      "Target: %s center=(%.1f, %.1f) area=%.1f", detection.label.c_str(), x,
      y_lb, detection.area);
  } else {
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

cv::Vec3d ImageProcessorNode::meanCenterColor(const cv::Mat &frame) const {
  if (frame.empty()) {
    return cv::Vec3d(0.0, 0.0, 0.0);
  }
  const cv::Rect sample_rect = sampleRect(frame);
  if (sample_rect.area() <= 0) {
    return cv::Vec3d(0.0, 0.0, 0.0);
  }

  cv::Mat sample = frame(sample_rect);
  const cv::Scalar mean = cv::mean(sample);
  return cv::Vec3d(mean[0], mean[1], mean[2]);
}

cv::Rect ImageProcessorNode::sampleRect(const cv::Mat &frame) const {
  if (frame.empty()) {
    return cv::Rect();
  }

  const int sample_x = 572;
  const int sample_y = 612;
  const int half = 5;

  const int x0 = std::clamp(sample_x - half, 0, frame.cols - 1);
  const int y0 = std::clamp(sample_y - half, 0, frame.rows - 1);
  const int x1 = std::clamp(sample_x + half - 1, 0, frame.cols - 1);
  const int y1 = std::clamp(sample_y + half - 1, 0, frame.rows - 1);

  if (x1 < x0 || y1 < y0) {
    return cv::Rect();
  }

  return cv::Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
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
  const double max_rg = std::max(r, g);
  if (b > max_rg + color_delta_) {
    return "blue";
  }
  return "unknown";
}

ImageProcessorNode::DetectionResult ImageProcessorNode::detectTarget(
  const cv::Mat &roi, const std::string &preferred) const {
  DetectionResult best;
  if (roi.empty()) {
    return best;
  }

  cv::Mat hsv;
  cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

  const int s_min = std::clamp(hsv_s_min_, 0, 255);
  const int v_min = std::clamp(hsv_v_min_, 0, 255);

  auto build_mask = [&](const std::string &label) -> cv::Mat {
    cv::Mat mask;
    if (label == "red") {
      cv::Mat mask1;
      cv::Mat mask2;
      cv::inRange(hsv, cv::Scalar(red_h_low1_, s_min, v_min),
                  cv::Scalar(red_h_high1_, 255, 255), mask1);
      cv::inRange(hsv, cv::Scalar(red_h_low2_, s_min, v_min),
                  cv::Scalar(red_h_high2_, 255, 255), mask2);
      cv::bitwise_or(mask1, mask2, mask);
    } else if (label == "blue") {
      cv::inRange(hsv, cv::Scalar(blue_h_low_, s_min, v_min),
                  cv::Scalar(blue_h_high_, 255, 255), mask);
    }
    return mask;
  };

  struct Marker {
    cv::Rect bbox;
    cv::Point2f center;
  };

  auto collect_markers = [&](const cv::Mat &mask) {
    std::vector<Marker> markers;
    if (mask.empty()) {
      return markers;
    }
    cv::Mat cleaned = mask;
    const int k = std::max(1, morph_kernel_);
    if (k > 1) {
      const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(k, k));
      cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN, kernel);
      cv::morphologyEx(cleaned, cleaned, cv::MORPH_CLOSE, kernel);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cleaned, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    for (const auto &contour : contours) {
      const double area = cv::contourArea(contour);
      if (area < static_cast<double>(marker_min_area_) ||
          area > static_cast<double>(marker_max_area_)) {
        continue;
      }
      cv::Rect rect = cv::boundingRect(contour);
      Marker marker;
      marker.bbox = rect;
      marker.center = cv::Point2f(rect.x + rect.width * 0.5f,
                                  rect.y + rect.height * 0.5f);
      markers.push_back(marker);
    }
    return markers;
  };

  auto evaluate_marker_pairs = [&](const std::vector<Marker> &markers,
                                   const std::string &label) {
    if (markers.size() < 2) {
      return;
    }
    for (size_t i = 0; i + 1 < markers.size(); ++i) {
      for (size_t j = i + 1; j < markers.size(); ++j) {
        const float dy = std::abs(markers[i].center.y - markers[j].center.y);
        if (dy > static_cast<float>(pair_y_tol_)) {
          continue;
        }
        const float dx = std::abs(markers[i].center.x - markers[j].center.x);
        if (dx < static_cast<float>(pair_min_dx_)) {
          continue;
        }

        cv::Rect rect = markers[i].bbox | markers[j].bbox;
        const int pad = std::max(0, bbox_pad_);
        rect.x = std::max(0, rect.x - pad);
        rect.y = std::max(0, rect.y - pad);
        rect.width = std::min(roi.cols - rect.x, rect.width + pad * 2);
        rect.height = std::min(roi.rows - rect.y, rect.height + pad * 2);

        const double area = static_cast<double>(rect.area());
        if (!best.found || area > best.area) {
          best.found = true;
          best.area = area;
          best.bbox = rect;
          best.center = cv::Point2f(rect.x + rect.width * 0.5f,
                                    rect.y + rect.height * 0.5f);
          best.label = label;
        }
      }
    }
  };

  auto detect_color = [&](const std::string &label) {
    const cv::Mat mask = build_mask(label);
    const std::vector<Marker> markers = collect_markers(mask);
    evaluate_marker_pairs(markers, label);
  };

  if (preferred == "red") {
    detect_color("blue");
  } else if (preferred == "blue") {
    detect_color("red");
  } else {
    detect_color("red");
    detect_color("blue");
  }

  return best;
}

}  // namespace cilent

