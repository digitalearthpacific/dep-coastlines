#!/usr/bin/env bash

pip install "git+https://github.com/digitalearthpacific/dep-tools.git@dev/modularize"
pip install "git+https://github.com/jessjaco/azure-logger.git"
pip install -I "git+https://github.com/jessjaco/odc-stac.git@exception-fix"
python "$@"
