cd $HOME
sudo apt update && sudo apt upgrade -y
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd $HOME/BearCar
uv sync --group server

# Coloring output
PURPLE='\033[0;35m'
NC='\033[0m' # No Color
echo -e "You may start BearCar environemnt now: ${PURPLE}gobear${NC}"
