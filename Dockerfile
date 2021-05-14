FROM rocker/binder:3.6.0

USER root
ENV PATH /opt/conda/bin:$PATH
# install the notebook package
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache notebook && \
    apt-get update

# create user with a home directory
ARG NB_USER
ARG NB_UID

# ENV USER ${NB_USER}
# ENV HOME /home/${NB_USER}

USER root
COPY . ${HOME}
RUN chown -R ${NB_USER} ${HOME}
# RUN chown -R ${NB_USER} /opt/conda
USER ${NB_USER}
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && \
    apt-get install -f -y --no-install-recommends texlive-latex-base && \
    apt-get install -f -y texlive-pictures && \
    apt-get install -f -y vim
