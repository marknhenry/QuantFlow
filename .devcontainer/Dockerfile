FROM ubuntu:latest

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
EXPOSE 8501

# Basic Installation and fixing timezone
RUN apt-get update  \
    && apt-get install -yq tzdata sudo jq \
    && ln -fs /usr/share/zoneinfo/Asia/Dubai /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && apt-get install -y wget graphviz libgraphviz-dev unzip git sudo \
    && rm -rf /var/lib/apt/lists/*

# Github CLI Installation
RUN type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)

RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && sudo apt update \
    && sudo apt install gh -y

# Miniconda Installation
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
    && mkdir /root/.conda \
    # && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && bash Miniconda3-latest-Linux-aarch64.sh -b \
    # && rm -f Miniconda3-latest-Linux-x86_64.sh 
    && rm -f Miniconda3-latest-Linux-aarch64.sh 

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
&& conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main

#Setting up Base Environment for work
RUN conda update -n base -c defaults conda -y \
    && conda init bash \
    && conda install -y numpy pandas matplotlib scipy sympy autopep8 requests seaborn numba scikit-learn \
    && conda install conda-forge::yfinance -y \
    # && conda install -c conda-forge notebook \
    && conda install -c conda-forge jupyterlab jupyterlab-drawio \
    && pip install jupyterlab-code-formatter black isort \
    && conda clean -a 
    # && apt update && apt install -y git-lfs \
    # && git lfs install \
    # && git lfs update --force 
    
