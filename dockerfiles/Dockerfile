
FROM ubuntu:20.04

ENV TZ=Etc/UTC
ARG DEBIAN_FRONTEND=noninteractive
ARG ROOT_PWD=root
ARG APT_PKG=apt_packages.txt
ARG PIP_PKG=pip_requirements.txt

USER root
# Set root password
RUN echo 'root:${ROOT_PWD}' | chpasswd

# Install
WORKDIR /tmp

# APT packages 
RUN apt update 
COPY ./${APT_PKG} ./
RUN xargs apt install \
    --yes \
    --no-install-recommends\
    < ${APT_PKG}

# Python packages
COPY ./${PIP_PKG} ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install \
    --no-cache-dir \
    --requirement \
    ${PIP_PKG}

# Clean up
RUN apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

CMD bash 