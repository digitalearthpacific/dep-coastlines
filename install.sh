#!/usr/bin/env bash

pip install -e ./dep-tools
pip install -e ./deafrica-coastlines
pip install -r ./pyfes/requirements/dev.txt
pip install -e ./pyfes

# tools installed above are not correct versions
pip install -e ./dea-notebooks/Tools
