#ifndef CILENT_SERIAL_SENDER_H
#define CILENT_SERIAL_SENDER_H

#include <string>

namespace cilent {

class SerialSender {
 public:
  explicit SerialSender(std::string device = "/dev/pts/8");
  ~SerialSender();

  bool open();
  void close();
  bool isOpen() const;

  bool sendYawDegrees(float degrees);
  bool sendFire();

 private:
  bool configurePort();

  std::string device_;
  int fd_;
};

}  // namespace cilent

#endif  // CILENT_SERIAL_SENDER_H
