#ifndef CILENT_KALMAN_FILTER_H
#define CILENT_KALMAN_FILTER_H

#include <Eigen/Dense>

namespace cilent {

class KalmanFilterCV2D {
 public:
  KalmanFilterCV2D(double dt, double process_noise_accel, double meas_noise_pos);

  void setDeltaT(double dt);
  void setProcessNoise(double sigma_a);
  void setMeasurementNoise(double sigma_z);

  void reset(double x, double y, double vx = 0.0, double vy = 0.0);

  void predict();
  void update(double meas_x, double meas_y);

  Eigen::Vector2d position() const;
  Eigen::Vector2d velocity() const;
  Eigen::Vector2d predictAhead(double lead_time) const;

 private:
  void updateTransition();
  void updateProcessNoise();

  double dt_;
  double sigma_a_;
  double sigma_z_;

  Eigen::Matrix4d F_;
  Eigen::Matrix<double, 2, 4> H_;
  Eigen::Matrix4d Q_;
  Eigen::Matrix2d R_;
  Eigen::Matrix4d P_;
  Eigen::Matrix4d I_;
  Eigen::Vector4d x_;
};

}  // namespace cilent

#endif  // CILENT_KALMAN_FILTER_H
