#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

namespace {

std::atomic<bool> g_running{true};

void onSignal(int) {
  g_running.store(false);
}

bool writeAll(int fd, const uint8_t* data, size_t size) {
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

bool configurePort(int fd) {
  termios tio{};
  if (tcgetattr(fd, &tio) != 0) {
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

  return tcsetattr(fd, TCSANOW, &tio) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string device = "/dev/pts/8";
  int interval_ms = 50;

  if (argc >= 2) {
    device = argv[1];
  }
  if (argc >= 3) {
    interval_ms = std::atoi(argv[2]);
    if (interval_ms < 0) {
      interval_ms = 0;
    }
  }

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  const int fd = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
  if (fd < 0) {
    std::cerr << "[fire_tester] failed to open " << device << std::endl;
    return 1;
  }

  if (!configurePort(fd)) {
    std::cerr << "[fire_tester] failed to configure serial " << device << std::endl;
    ::close(fd);
    return 2;
  }

  std::cout << "[fire_tester] sending fire frame 0x02 to " << device
            << " every " << interval_ms << " ms" << std::endl;

  const uint8_t fire = 0x02;
  uint64_t count = 0;
  while (g_running.load()) {
    if (!writeAll(fd, &fire, 1)) {
      std::cerr << "[fire_tester] write failed" << std::endl;
      ::close(fd);
      return 3;
    }

    ++count;
    if (count % 20 == 0) {
      std::cout << "[fire_tester] sent " << count << " fire frames" << std::endl;
    }

    if (interval_ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
  }

  std::cout << "[fire_tester] stopped, total=" << count << std::endl;
  ::close(fd);
  return 0;
}
