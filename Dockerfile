FROM python:3.7-slim
# install the notebook package
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache notebook && \
    apt-get update

# create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER} && \
    apt-get update && \
    apt-get install -f -y --no-install-recommends texlive-latex-base && \
    apt-get install -f -y texlive-pictures && \
    apt-get install -f -y vim
WORKDIR ${HOME}
USER ${USER}
