FROM ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        cargo \
        coreutils \
        g++ \
        gcc \
        git \
        golang-go \
        make \
        node-typescript \
        nodejs \
        npm \
        openjdk-21-jdk-headless \
        python3 \
        python3-pip \
        python3-venv \
        rustc \
    && rm -rf /var/lib/apt/lists/* \
    && command -v bash cargo g++ gcc git go java javac node npm python3 rustc timeout tsc >/dev/null \
    && test "$(git --version)" = "git version 2.43.0"

LABEL org.hone-subnet.execution-profile="repo-polyglot-v1"

USER 65534:65534
WORKDIR /tmp
