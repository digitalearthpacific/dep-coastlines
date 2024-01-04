#!/usr/bin/env bash

pip install "git+https://github.com/digitalearthpacific/dep-tools.git@develop"
pip install "git+https://github.com/jessjaco/azure-logger.git"
pip install "git+https://github.com/digitalearthpacific/dep-grid.git@cleanup"
python "$@"
