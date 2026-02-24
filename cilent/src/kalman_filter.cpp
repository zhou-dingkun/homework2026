#include "cilent/kalman_filter.h"

#include <algorithm>

namespace cilent {

KalmanFilterCV2D::KalmanFilterCV2D(double dt, double process_noise_accel,
                                   double meas_noise_pos)
    : dt_(dt),
      sigma_a_(process_noise_accel),
      sigma_z_(meas_noise_pos),
      F_(Eigen::Matrix4d::Identity()),
      H_(Eigen::Matrix<double, 2, 4>::Zero()),
      Q_(Eigen::Matrix4d::Zero()),
      R_(Eigen::Matrix2d::Identity()),
      P_(Eigen::Matrix4d::Identity()),
      I_(Eigen::Matrix4d::Identity()),
      x_(Eigen::Vector4d::Zero()) {
  H_(0, 0) = 1.0;
  H_(1, 1) = 1.0;

  setDeltaT(dt_);
  setProcessNoise(sigma_a_);
  setMeasurementNoise(sigma_z_);

  P_.setIdentity();
  P_(2, 2) = 10.0;
  P_(3, 3) = 10.0;
}

void KalmanFilterCV2D::setDeltaT(double dt) {
  if (dt <= 0.0) {
    return;
  }
  dt_ = dt;
  updateTransition();
  updateProcessNoise();
}

void KalmanFilterCV2D::setProcessNoise(double sigma_a) {
  if (sigma_a <= 0.0) {
    return;
  }
  sigma_a_ = sigma_a;
  updateProcessNoise();
}

void KalmanFilterCV2D::setMeasurementNoise(double sigma_z) {
  if (sigma_z <= 0.0) {
    return;
  }
  sigma_z_ = sigma_z;
  R_.setIdentity();
  R_ *= sigma_z_ * sigma_z_;
}

void KalmanFilterCV2D::reset(double x, double y, double vx, double vy) {
  x_ << x, y, vx, vy;
  P_.setIdentity();
  P_(2, 2) = 10.0;
  P_(3, 3) = 10.0;
}

void KalmanFilterCV2D::predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilterCV2D::update(double meas_x, double meas_y) {
  Eigen::Vector2d z;
  z << meas_x, meas_y;

  const Eigen::Vector2d y = z - H_ * x_;
  const Eigen::Matrix2d s = H_ * P_ * H_.transpose() + R_;
  const Eigen::Matrix<double, 4, 2> k = P_ * H_.transpose() * s.inverse();

  x_ = x_ + k * y;
  P_ = (I_ - k * H_) * P_;
}

Eigen::Vector2d KalmanFilterCV2D::position() const {
  return x_.head<2>();
}

Eigen::Vector2d KalmanFilterCV2D::velocity() const {
  return x_.tail<2>();
}

Eigen::Vector2d KalmanFilterCV2D::predictAhead(double lead_time) const {
  const double dt = std::max(0.0, lead_time);
  Eigen::Matrix4d f = Eigen::Matrix4d::Identity();
  f(0, 2) = dt;
  f(1, 3) = dt;

  const Eigen::Vector4d x_pred = f * x_;
  return x_pred.head<2>();
}

void KalmanFilterCV2D::updateTransition() {
  F_.setIdentity();
  F_(0, 2) = dt_;
  F_(1, 3) = dt_;
}

void KalmanFilterCV2D::updateProcessNoise() {
  const double dt2 = dt_ * dt_;
  const double dt3 = dt2 * dt_;
  const double dt4 = dt2 * dt2;
  const double q = sigma_a_ * sigma_a_;

  Q_.setZero();
  Q_(0, 0) = dt4 * 0.25 * q;
  Q_(0, 2) = dt3 * 0.5 * q;
  Q_(1, 1) = dt4 * 0.25 * q;
  Q_(1, 3) = dt3 * 0.5 * q;
  Q_(2, 0) = dt3 * 0.5 * q;
  Q_(2, 2) = dt2 * q;
  Q_(3, 1) = dt3 * 0.5 * q;
  Q_(3, 3) = dt2 * q;
}

}  // namespace cilent
