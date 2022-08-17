SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

conda env create -f $SCRIPT_DIR/../droplet_environment.yml
