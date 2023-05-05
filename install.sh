#!/usr/bin/env bash

pip install -e ./dep-tools
pip install -e ./deafrica-coastlines

# tools installed above are not correct versions
pip install -e ./dea-notebooks/Tools
# Missing dependency for dea-notebooks
pip install odc-geo

