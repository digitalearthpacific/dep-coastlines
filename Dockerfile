FROM mcr.microsoft.com/planetary-computer/python:latest AS base

ADD . /tmp/src
ADD requirements.txt /tmp/requirements.txt
RUN conda install gcc -y \
  && pip install --no-cache-dir --upgrade pip \
  && pip install -r /tmp/requirements.txt
