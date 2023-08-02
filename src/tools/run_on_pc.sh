#!/usr/bin/env bash

pip install "git+https://github.com/digitalearthpacific/dep-tools.git@dev/modularize"
pip install "git+https://github.com/jessjaco/azure-logger.git"
python $1
