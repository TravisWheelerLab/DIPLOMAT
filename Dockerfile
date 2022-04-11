FROM python:3.8-bullseye

RUN apt-get -q update \
    && apt-get -y install \
        build-essential \
        make \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data

RUN pip install pipenv==2022.4.8

ENV HOME=/data

VOLUME /data
WORKDIR /data

