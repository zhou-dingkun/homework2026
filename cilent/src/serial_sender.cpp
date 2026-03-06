#include "cilent/serial_sender.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstring>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

namespace cilent {

namespace {

bool toNativeFloatBytes(float value, std::array<uint8_t, 4> *out) {
  static_assert(sizeof(float) == 4, "float32 required");
  if (!out) {
    return false;
  }
  std::memcpy(out->data(), &value, sizeof(value));
  return true;
}

bool writeAll(int fd, const uint8_t *data, size_t size) {
  if (fd >= 0 && data && size > 0) {
    std::fprintf(stderr, "[serial tx] fd=%d len=%zu data=", fd, size);
    for (size_t i = 0; i < size; ++i) {
      std::fprintf(stderr, "%02X", static_cast<unsigned int>(data[i]));
      if (i + 1 < size) {
        std::fputc(' ', stderr);
      }
    }
    std::fputc('\n', stderr);
  }

  size_t sent = 0;
  while (sent < size) {
    const ssize_t rc = ::write(fd, data + sent, size - sent);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    sent += static_cast<size_t>(rc);
  }
  return true;
}

}  // namespace

SerialSender::SerialSender(std::string device)
  : device_(std::move(device)), fd_(-1) {}

SerialSender::~SerialSender() { close(); }

bool SerialSender::open() {
  if (fd_ >= 0) {
    return true;
  }
  fd_ = ::open(device_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
  if (fd_ < 0) {
    return false;
  }
  if (!configurePort()) {
    close();
    return false;
  }
  return true;
}

void SerialSender::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

bool SerialSender::isOpen() const { return fd_ >= 0; }

bool SerialSender::sendYawDegrees(float degrees) {
  if (!isOpen()) {
    return false;
  }
  const float clamped = std::clamp(degrees, -180.0f, 180.0f);
  std::array<uint8_t, 4> payload{};
  if (!toNativeFloatBytes(clamped, &payload)) {
    return false;
  }

  uint8_t frame[5] = {0x01, payload[0], payload[1], payload[2], payload[3]};
  return writeAll(fd_, frame, sizeof(frame));
}

bool SerialSender::sendFire() {
  if (!isOpen()) {
    return false;
  }
  const uint8_t frame = 0x02;
  return writeAll(fd_, &frame, 1);
}

bool SerialSender::configurePort() {
  termios tio{};
  if (tcgetattr(fd_, &tio) != 0) {
    return false;
  }

  cfmakeraw(&tio);
  cfsetispeed(&tio, B115200);
  cfsetospeed(&tio, B115200);

  tio.c_cflag |= (CLOCAL | CREAD);
  tio.c_cflag &= ~CRTSCTS;
  tio.c_cflag &= ~CSTOPB;
  tio.c_cflag &= ~PARENB;
  tio.c_cflag &= ~CSIZE;
  tio.c_cflag |= CS8;

  if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
    return false;
  }
  return true;
}

}  // namespace cilent
