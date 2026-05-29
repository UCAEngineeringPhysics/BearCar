cd $HOME
sudo apt update && sudo apt upgrade -y
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd $HOME/BearCar
uv venv --system-site-packages --python /usr/bin/python3
uv sync --group server --extra cu130
echo 'alias gobear="source $HOME/BearCar/.venv/bin/activate"' >>$HOME/.bashrc
echo 'eval "$(uv generate-shell-completion bash)"' >>$HOME/.bashrc
source $HOME/.bashrc

# Coloring output
PURPLE='\033[0;35m'
NC='\033[0m' # No Color
echo -e "You may start BearCar environemnt now: ${PURPLE}gobear${NC}"
