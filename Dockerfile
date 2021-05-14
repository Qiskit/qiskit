FROM rocker/binder:3.6.0

# add conda and other needed utilities based on https://hub.docker.com/r/continuumio/miniconda3/dockerfile and 
# https://hub.docker.com/r/rocker/binder/dockerfile
USER root
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache notebook && \
    apt-get update

RUN apt-get update && \
    apt-get install -y wget gzip bzip2 ca-certificates curl git && \
    apt-get purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -f -y --no-install-recommends texlive-latex-base && \
    apt-get install -f -y texlive-pictures && \
    apt-get install -f -y vim
    
# Copy repo into ${HOME}, make user own $HOME
USER root
COPY . ${HOME}
RUN chown -R ${NB_USER} ${HOME}
# RUN chown -R ${NB_USER} /opt/conda
USER ${NB_USER}

# ENV PATH /opt/conda/bin:$PATH
