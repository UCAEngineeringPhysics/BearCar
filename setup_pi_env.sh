cd $HOME
sudo apt update && sudo apt upgrade -y
echo 'eval "$(uv generate-shell-completion bash)"' >>~/.bashrc
source $HOME/.bashrc
