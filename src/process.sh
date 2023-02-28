#!/usr/bin/env bash

mkdir -p /var/share/tide_models
tar zxvf /tmp/src/data/fes2014.tar.gz -C /var/share/tide_models
rm /tmp/src/data/fes2014.tar.gz
python run_coastlines.py
