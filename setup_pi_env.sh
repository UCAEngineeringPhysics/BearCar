cd $HOME
sudo apt update && sudo apt upgrade -y
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd $HOME/BearCart
uv venv --system-site-packages
echo 'alias gobear="source $HOME/BearCart/.venv/bin/activate"' >>$HOME/.bashrc
echo 'eval "$(uv generate-shell-completion bash)"' >>$HOME/.bashrc
uv sync
source $HOME/.bashrc

# Coloring output
PURPLE='\033[0;35m'
NC='\033[0m' # No Color
echo -e 'You may start BearCart environemnt now: ${PURPLE}gobear${NC}'
