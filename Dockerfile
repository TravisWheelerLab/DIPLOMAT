FROM ubuntu:20.04

RUN apt-get -q update && \
    apt-get -y install \
        build-essential \
        make && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /code

VOLUME /code
WORKDIR /code

