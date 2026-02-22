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

# Start the game in background.
"$SCRIPT_DIR/homework2026.sh" &
GAME_PID=$!

# Run the auto-aim node.
ros2 run cilent cilent_main

# If the node exits, stop the game.
kill "$GAME_PID" >/dev/null 2>&1 || true
