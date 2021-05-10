FROM python:3.7-slim
ENV NODE_ENV=production

RUN apt update -y && apt install -y locales locales-all git curl
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

WORKDIR /usr/src/app

RUN apt install -y gcc

COPY . ner/
RUN cd ner && \
    pip install -r requirements.txt

RUN curl -L -O curl -L -O https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz && \
    tar xvf openjdk-11.0.2_linux-x64_bin.tar.gz && \
    rm openjdk-11.0.2_linux-x64_bin.tar.gz
ENV JAVA_HOME=/usr/src/app/jdk-11.0.2

COPY model.bin ner/
