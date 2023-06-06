#1/usr/bin/env bash

files = $(az storage fs file list -f output --path coastlines-vector --recursive -o tsv | cut -f9 |sed 's/^/\/vsiaz\/output\//' | tr '\n' ' ')

