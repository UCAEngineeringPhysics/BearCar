cd $HOME
sudo apt update && sudo apt upgrade -y
curl -LsSf https://astral.sh/uv/install.sh | sh
cd $HOME/BearCart/ && uv venv
echo 'eval "$(uv generate-shell-completion bash)"' >>$HOME/.bashrc
echo 'alias gobc="source $HOME/BearCart/.venv/bin/activate"' >>$HOME/.bashrc
source $HOME/.bashrc
