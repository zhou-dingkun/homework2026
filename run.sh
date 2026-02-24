#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${ROS_SETUP:-}" && -f "$ROS_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
elif [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
elif [[ -f /opt/ros/foxy/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/foxy/setup.bash
fi

cd "$SCRIPT_DIR"
if [[ -f install/setup.bash ]]; then
  # shellcheck disable=SC1091
  source install/setup.bash
fi

serial_device=""
passthrough_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--serial)
      if [[ -n "${2:-}" ]]; then
        serial_device="$2"
        shift 2
      else
        echo "Error: --serial requires a device path" >&2
        exit 1
      fi
      ;;
    --)
      shift
      passthrough_args+=("$@")
      break
      ;;
    *)
      passthrough_args+=("$1")
      shift
      ;;
  esac
done

ros_args=()
if [[ -n "$serial_device" ]]; then
  ros_args+=(--ros-args -p "serial_device:=${serial_device}")
fi

# Run the auto-aim node without starting the game server.
ros2 run cilent cilent_main "${ros_args[@]}" "${passthrough_args[@]}"
