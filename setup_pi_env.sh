cd $HOME
sudo apt update && sudo apt upgrade -y
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
echo 'eval "$(uv generate-shell-completion bash)"' >>$HOME/.bashrc
echo 'alias gobear="source $HOME/BearCart/.venv/bin/activate"' >>$HOME/.bashrc
source $HOME/.bashrc
cd $HOME/BearCart
uv venv --system-site-packages
uv sync
echo 'You may start BearCart environemnt now'
