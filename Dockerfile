FROM jupyter/datascience-notebook 
USER root
RUN apt update && apt install -y zsh nano emacs
USER jovyan
RUN  sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
USER root
WORKDIR /
