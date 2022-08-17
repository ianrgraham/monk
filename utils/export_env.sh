#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

conda env export | grep -v "^prefix: " > $SCRIPT_DIR/../droplet_environment.yml
