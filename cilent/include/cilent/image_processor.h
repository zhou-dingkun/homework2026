#ifndef CILENT_IMAGE_PROCESSOR_H
#define CILENT_IMAGE_PROCESSOR_H

#include <string>
#include <vector>

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
    std::vector<cv::Rect> boxes;
    double area = 0.0;
    std::string label;
  };

  struct ImageContext {
    cv_bridge::CvImageConstPtr cv_ptr;
    cv::Rect crop;
    cv::Vec3d mean_bgr;
    std::string color;
    DetectionResult detection;
  };

 protected:
  virtual void onImage(const sensor_msgs::msg::Image::SharedPtr msg);
  bool buildImageContext(const sensor_msgs::msg::Image::SharedPtr &msg,
                         ImageContext &ctx);
  void maybeSaveDebugImages(const cv::Mat &frame, const cv::Rect &crop,
                            const DetectionResult &detection,
                            const cv::Point2f *aim_point = nullptr,
                            const std::vector<cv::Point2f> *all_aim_points =
                              nullptr,
                            const std::vector<int> *all_aim_track_ids =
                              nullptr);
  virtual cv::Rect computeCropRect(int width, int height) const;
  virtual cv::Vec3d meanCenterColor(const cv::Mat &frame) const;
  virtual std::string classifyColor(const cv::Vec3d &bgr) const;
  virtual DetectionResult detectTarget(const cv::Mat &roi,
                                       const std::string &preferred) const;
  cv::Rect sampleRect(const cv::Mat &frame) const;

  int expected_width_;
  int expected_height_;
  int crop_width_;
  int crop_height_;
  int center_window_;
  int sample_window_;
  int sample_offset_x_;
  int sample_offset_y_;
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
  int pair_max_dx_;
  int bbox_pad_;

  std::string debug_save_path_;
  int debug_save_interval_ms_;
  bool debug_save_overlay_;
  rclcpp::Time last_debug_save_time_;
  bool debug_save_frames_;
  std::string debug_frames_dir_;
  std::string debug_frames_dir_resolved_;
  int debug_frames_interval_ms_;
  uint64_t debug_frame_index_;
  rclcpp::Time last_debug_frame_time_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};

}  // namespace cilent

#endif  // CILENT_IMAGE_PROCESSOR_H
