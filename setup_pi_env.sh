cd $HOME
sudo apt update && sudo apt upgrade -y
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH=$PATH:$HOME/.local/bin' >>$HOME/.bashrc
echo 'eval "$(uv generate-shell-completion bash)"' >>$HOME/.bashrc
# cd $HOME/BearCart/ && uv venv
# echo 'alias gobc="source $HOME/BearCart/.venv/bin/activate"' >>$HOME/.bashrc
source $HOME/.bashrc
cd $HOME/BearCart/ && uv sync
echo 'You may start BearCart environemnt now'
