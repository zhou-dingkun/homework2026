#include "cilent/image_processor.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace cilent {

ImageProcessorNode::ImageProcessorNode(const rclcpp::NodeOptions &options)
  : rclcpp::Node("image_processor", options) {
  expected_width_ = declare_parameter<int>("expected_width", 1152);
  expected_height_ = declare_parameter<int>("expected_height", 648);
  crop_width_ = declare_parameter<int>("crop_width", expected_width_);
  crop_height_ = declare_parameter<int>("crop_height", expected_height_);
  center_window_ = declare_parameter<int>("center_window", 7);
  sample_window_ = declare_parameter<int>("sample_window", 10);
  sample_offset_x_ = declare_parameter<int>("sample_offset_x", 0);
  sample_offset_y_ = declare_parameter<int>("sample_offset_y", 20);
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
  marker_max_area_ = declare_parameter<int>("marker_max_area", 50000);
  pair_y_tol_ = declare_parameter<int>("pair_y_tol", 10);
  pair_min_dx_ = declare_parameter<int>("pair_min_dx", 20);
  pair_max_dx_ = declare_parameter<int>("pair_max_dx", 120);
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
    declare_parameter<int>("debug_frames_interval_ms", 1000);
  if (debug_frames_interval_ms_ < 1000) {
    RCLCPP_WARN(get_logger(),
                "debug_frames_interval_ms=%d is too small, force to 1000ms",
                debug_frames_interval_ms_);
    debug_frames_interval_ms_ = 1000;
  }
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
  ImageContext ctx;
  if (!buildImageContext(msg, ctx)) {
    return;
  }

  const cv::Mat &frame = ctx.cv_ptr->image;
  const cv::Rect &crop = ctx.crop;
  const cv::Vec3d &mean_bgr = ctx.mean_bgr;
  const std::string &color = ctx.color;
  const DetectionResult &detection = ctx.detection;

  maybeSaveDebugImages(frame, crop, detection);

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

bool ImageProcessorNode::buildImageContext(
  const sensor_msgs::msg::Image::SharedPtr &msg, ImageContext &ctx) {
  if (!msg) {
    return false;
  }

  const uint64_t required_bytes =
    static_cast<uint64_t>(msg->step) * static_cast<uint64_t>(msg->height);
  if (msg->data.size() < required_bytes) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Malformed image message: data.size=%zu < step*height=%llu",
      msg->data.size(), static_cast<unsigned long long>(required_bytes));
    return false;
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
    return false;
  } catch (const cv::Exception &ex) {
    RCLCPP_ERROR(get_logger(), "OpenCV exception in cv_bridge: %s", ex.what());
    return false;
  } catch (const std::exception &ex) {
    RCLCPP_ERROR(get_logger(), "Exception in cv_bridge: %s", ex.what());
    return false;
  }

  const cv::Mat &frame = cv_ptr->image;
  if (frame.empty()) {
    return false;
  }

  constexpr int64_t kMaxProcessPixels = 40000000LL;  // 40 MP
  const int64_t frame_pixels =
    static_cast<int64_t>(frame.cols) * static_cast<int64_t>(frame.rows);
  if (frame.cols <= 0 || frame.rows <= 0 || frame_pixels <= 0) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Invalid cv::Mat shape: cols=%d rows=%d",
                         frame.cols, frame.rows);
    return false;
  }
  if (frame_pixels > kMaxProcessPixels) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Image too large: %dx%d (%lld pixels), skip to avoid OOM",
      frame.cols, frame.rows, static_cast<long long>(frame_pixels));
    return false;
  }

  try {
    ctx.crop = computeCropRect(frame.cols, frame.rows);
    if (ctx.crop.width <= 0 || ctx.crop.height <= 0) {
      return false;
    }

    const cv::Mat roi = frame(ctx.crop);
    ctx.mean_bgr = meanCenterColor(frame);
    ctx.color = classifyColor(ctx.mean_bgr);
    ctx.detection = detectTarget(roi, ctx.color);
    ctx.cv_ptr = cv_ptr;
  } catch (const cv::Exception &ex) {
    RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                          "OpenCV exception in processing: %s", ex.what());
    return false;
  }

  return true;
}

void ImageProcessorNode::maybeSaveDebugImages(
  const cv::Mat &frame, const cv::Rect &crop,
  const DetectionResult &detection, const cv::Point2f *aim_point,
  const std::vector<cv::Point2f> *all_aim_points,
  const std::vector<int> *all_aim_track_ids) {
  std::vector<cv::Rect> detection_rects;
  detection_rects.reserve(detection.boxes.size());
  for (const auto &box : detection.boxes) {
    cv::Rect rect = box;
    rect.x += crop.x;
    rect.y += crop.y;
    rect &= cv::Rect(0, 0, frame.cols, frame.rows);
    if (rect.area() > 0) {
      detection_rects.push_back(rect);
    }
  }

  cv::Rect best_detection_rect;
  if (detection.found) {
    best_detection_rect = detection.bbox;
    best_detection_rect.x += crop.x;
    best_detection_rect.y += crop.y;
    best_detection_rect &= cv::Rect(0, 0, frame.cols, frame.rows);
  }

  auto draw_annotations = [&](cv::Mat &img) {
    cv::Rect crop_rect = crop & cv::Rect(0, 0, img.cols, img.rows);
    if (crop_rect.area() > 0) {
      cv::rectangle(img, crop_rect, cv::Scalar(255, 255, 0), 2);
    }

    const cv::Rect sample_rect = sampleRect(img);
    if (sample_rect.area() > 0) {
      cv::rectangle(img, sample_rect, cv::Scalar(0, 255, 0), 2);
    }

    for (const auto &rect : detection_rects) {
      cv::rectangle(img, rect, cv::Scalar(0, 165, 255), 2);
    }

    if (detection.found && best_detection_rect.area() > 0) {
      cv::rectangle(img, best_detection_rect, cv::Scalar(0, 255, 255), 2);
      cv::putText(img, detection.label,
                  cv::Point(best_detection_rect.x,
                            std::max(20, best_detection_rect.y - 6)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  cv::Scalar(0, 255, 255), 2);
    } else {
      cv::putText(img, "NO TARGET",
                  cv::Point(std::max(10, crop_rect.x + 4),
                            std::max(24, crop_rect.y + 24)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.7,
                  cv::Scalar(0, 0, 255), 2);
    }

    if (all_aim_points) {
      for (size_t i = 0; i < all_aim_points->size(); ++i) {
        cv::Point2f p = (*all_aim_points)[i];
        p.x = std::clamp(p.x, 0.0f, static_cast<float>(img.cols - 1));
        p.y = std::clamp(p.y, 0.0f, static_cast<float>(img.rows - 1));
        const cv::Point pi(static_cast<int>(std::lround(p.x)),
                           static_cast<int>(std::lround(p.y)));
        cv::circle(img, pi, 4, cv::Scalar(255, 128, 0), 2);

        std::ostringstream label;
        if (all_aim_track_ids && i < all_aim_track_ids->size()) {
          label << "T" << (*all_aim_track_ids)[i] << "(" << pi.x << ","
                << pi.y << ")";
        } else {
          label << "P" << i << "(" << pi.x << "," << pi.y << ")";
        }
        cv::putText(img, label.str(),
                    cv::Point(pi.x + 6, std::max(20, pi.y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(255, 128, 0), 1);
      }
    }

    if (aim_point) {
      cv::Point2f p = *aim_point;
      p.x = std::clamp(p.x, 0.0f, static_cast<float>(img.cols - 1));
      p.y = std::clamp(p.y, 0.0f, static_cast<float>(img.rows - 1));
      const cv::Point pi(static_cast<int>(std::lround(p.x)),
                         static_cast<int>(std::lround(p.y)));
      cv::circle(img, pi, 6, cv::Scalar(255, 0, 255), 2);
      cv::putText(img, "LEAD", cv::Point(pi.x + 8, std::max(20, pi.y - 8)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 0, 255), 2);
    }
  };

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
        try {
          cv::Mat debug_frame = frame.clone();
          draw_annotations(debug_frame);

          std::ostringstream name;
          name << debug_frames_dir_resolved_ << "/frame_"
               << std::setfill('0')
               << std::setw(6) << debug_frame_index_++ << ".png";
          if (!cv::imwrite(name.str(), debug_frame)) {
            RCLCPP_WARN(get_logger(), "Failed to write debug frame to %s",
                        name.str().c_str());
          } else {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "Saved debug frame to %s",
                                 name.str().c_str());
          }
        } catch (const cv::Exception &ex) {
          RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                                "OpenCV exception in frame save: %s",
                                ex.what());
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
      try {
        cv::Mat debug_frame = frame.clone();
        draw_annotations(debug_frame);
        if (!cv::imwrite(debug_save_path_, debug_frame)) {
          RCLCPP_WARN(get_logger(), "Failed to write debug overlay to %s",
                      debug_save_path_.c_str());
        } else {
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                               "Saved debug overlay to %s",
                               debug_save_path_.c_str());
        }
      } catch (const cv::Exception &ex) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                              "OpenCV exception in overlay save: %s",
                              ex.what());
      }
      last_debug_save_time_ = now;
    }
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

  const int window = std::max(1, sample_window_);
  const int half = window / 2;
  const int sample_x = frame.cols / 2 + sample_offset_x_;
  const int sample_y = frame.rows - 1 - half - sample_offset_y_;

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

  std::vector<std::string> enemy_colors;
  if (preferred == "red") {
    enemy_colors = {"blue"};
  } else if (preferred == "blue") {
    enemy_colors = {"red"};
  } else {
    // 无法判定自车颜色时，保守地把红蓝都视为敌方
    enemy_colors = {"red", "blue"};
  }

  double best_score = -std::numeric_limits<double>::infinity();

  for (const auto &label : enemy_colors) {
    cv::Mat mask = build_mask(label);
    if (mask.empty()) {
      continue;
    }

    const int k = std::max(1, morph_kernel_);
    if (k > 1) {
      const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(k, k));
      cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
      cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> markers;
    markers.reserve(contours.size());

    for (const auto &contour : contours) {
      const double contour_area = cv::contourArea(contour);
      if (contour_area < static_cast<double>(marker_min_area_) ||
          contour_area > static_cast<double>(marker_max_area_)) {
        continue;
      }

      cv::Rect rect = cv::boundingRect(contour);
      const int pad = std::max(0, bbox_pad_);
      rect.x = std::max(0, rect.x - pad);
      rect.y = std::max(0, rect.y - pad);
      rect.width = std::min(roi.cols - rect.x, rect.width + pad * 2);
      rect.height = std::min(roi.rows - rect.y, rect.height + pad * 2);
      if (rect.width <= 0 || rect.height <= 0) {
        continue;
      }

      markers.push_back(rect);
    }

    if (markers.size() < 2) {
      continue;
    }

    for (size_t i = 0; i < markers.size(); ++i) {
      const cv::Rect &a = markers[i];
      const float ax = a.x + a.width * 0.5f;
      const float ay = a.y + a.height * 0.5f;
      for (size_t j = i + 1; j < markers.size(); ++j) {
        const cv::Rect &b = markers[j];
        const float bx = b.x + b.width * 0.5f;
        const float by = b.y + b.height * 0.5f;

        const double y_diff = std::abs(static_cast<double>(ay - by));
        if (y_diff > static_cast<double>(std::max(1, pair_y_tol_))) {
          continue;
        }

        const double dx = std::abs(static_cast<double>(ax - bx));
        if (dx < static_cast<double>(std::max(1, pair_min_dx_))||
            dx > static_cast<double>(std::max(1, pair_max_dx_))) {
          continue;
        }

        const double h1 = static_cast<double>(a.height);
        const double h2 = static_cast<double>(b.height);
        const double h_ratio = std::max(h1, h2) / std::max(1.0, std::min(h1, h2));
        if (h_ratio > 2.0) {
          continue;
        }

        cv::Rect armor = a | b;
        const int pad = std::max(0, bbox_pad_);
        armor.x = std::max(0, armor.x - pad);
        armor.y = std::max(0, armor.y - pad);
        armor.width = std::min(roi.cols - armor.x, armor.width + pad * 2);
        armor.height = std::min(roi.rows - armor.y, armor.height + pad * 2);
        if (armor.width <= 0 || armor.height <= 0) {
          continue;
        }

        best.boxes.push_back(armor);

        const double area = static_cast<double>(armor.area());
        const double score = area - y_diff * 10.0 - std::abs(h1 - h2) * 8.0;
        if (!best.found || score > best_score) {
          best_score = score;
          best.found = true;
          best.area = area;
          best.bbox = armor;
          best.center = cv::Point2f(armor.x + armor.width * 0.5f,
                                    armor.y + armor.height * 0.5f);
          best.label = label;
        }
      }
    }
  }

  return best;
}

}  // namespace cilent

