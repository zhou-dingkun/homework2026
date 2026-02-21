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

 protected:
  virtual void onImage(const sensor_msgs::msg::Image::SharedPtr msg);
  virtual cv::Rect computeCropRect(int width, int height) const;
  virtual cv::Vec3d meanCenterColor(const cv::Mat &roi) const;
  virtual std::string classifyColor(const cv::Vec3d &bgr) const;

  int expected_width_;
  int expected_height_;
  int crop_width_;
  int crop_height_;
  int center_window_;
  int color_delta_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};

}  // namespace cilent

#endif  // CILENT_IMAGE_PROCESSOR_H
