cd $HOME
sudo apt update && sudo apt upgrade -y
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'eval "$(uv generate-shell-completion bash)"' >>~/.bashrc
source $HOME/.bashrc
