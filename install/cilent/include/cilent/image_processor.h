#ifndef CILENT_IMAGE_PROCESSOR_H
#define CILENT_IMAGE_PROCESSOR_H

#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace cilent {

class ImageProcessorNode : public rclcpp::Node {
 public:
  explicit ImageProcessorNode(const rclcpp::NodeOptions &options =
                                  rclcpp::NodeOptions());

  struct DetectionResult {
    bool found = false;
    cv::Point2f center;
    cv::Rect bbox;
    double area = 0.0;
    std::string label;
  };

 protected:
  virtual void onImage(const sensor_msgs::msg::Image::SharedPtr msg);
  virtual cv::Rect computeCropRect(int width, int height) const;
  virtual cv::Vec3d meanCenterColor(const cv::Mat &roi) const;
  virtual std::string classifyColor(const cv::Vec3d &bgr) const;
  virtual DetectionResult detectTarget(const cv::Mat &roi,
                                       const std::string &preferred) const;

  int expected_width_;
  int expected_height_;
  int crop_width_;
  int crop_height_;
  int center_window_;
  int color_delta_;

  int hsv_s_min_;
  int hsv_v_min_;
  int red_h_low1_;
  int red_h_high1_;
  int red_h_low2_;
  int red_h_high2_;
  int blue_h_low_;
  int blue_h_high_;
  int morph_kernel_;
  int marker_min_area_;
  int marker_max_area_;
  int pair_y_tol_;
  int pair_min_dx_;
  int bbox_pad_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};

}  // namespace cilent

#endif  // CILENT_IMAGE_PROCESSOR_H
