#!/usr/bin/env bash

pip install "git+https://github.com/digitalearthpacific/dep-tools.git@coastline-fixes"
pip install "git+https://github.com/jessjaco/azure-logger.git@0376f92"
pip install --force-reinstall --no-deps "git+https://github.com/jessjaco/odc-stac.git@exception-fix"
python "$@"
