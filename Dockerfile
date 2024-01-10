FROM mcr.microsoft.com/planetary-computer/python:latest AS base

USER root

ADD . /tmp/dep-coastlines
WORKDIR /tmp
RUN pip install --no-cache-dir --upgrade pip && pip install ./dep-coastlines

ADD . /code
WORKDIR /code

