#!/bin/sh
echo "!!!"
echo "WARNING: currently, this script only work with Raspberry Pi OS on Raspberry Pi 4/5 or Ubuntu/Debian on x86_64"
echo "!!!"

# Determine CPU architecture
ARCH=$(uname -m)
echo "$ARCH architecture detected."
case "$ARCH" in
aarch64)
  BEARCAR_ENV_TYPE="pi"
  ;;
x86_64)
  BEARCAR_ENV_TYPE="server"
  ;;
*)
  # Embed the unsupported architecture in the error message
  echo "Error: Architecture '$ARCH' is not supported."
  # Terminate the script with an error code
  exit 1
  ;;
esac
echo "BearCar environemnt type: $BEARCAR_ENV_TYPE"

cd $HOME
# Install uv
sudo apt update && sudo apt upgrade -y
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
echo 'eval "$(uv generate-shell-completion bash)"' >>$HOME/.bashrc
# Setup project env
cd $HOME/BearCar
uv venv --system-site-packages --python /usr/bin/python3
case "$BEARCAR_ENV_TYPE" in
pi)
  uv sync --group pi --extra cpu
  ;;
x86_64)
  uv sync --group server --extra cu130
  ;;
*)
  echo "BearCar environemnt type not found"
  ;;
esac
echo 'alias gobear="source $HOME/BearCar/.venv/bin/activate"' >>$HOME/.bashrc
# Output result
PURPLE='\033[0;35m' # Coloring output
NC='\033[0m'        # No Color
echo -e "You may start BearCar environemnt now: ${PURPLE}gobear${NC}"
source $HOME/.bashrc
