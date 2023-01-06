#!/usr/bin/env bash

mkdir -p /var/share/tide_models
tar zxvf /src/data/fes2014.tar.gz -C /var/share/tide_models
rm /src/data/fes2014.tar.gz
python /src/run_coastlines.py

