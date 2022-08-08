FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

ENV PATH="/opt/conda/bin:${PATH}"
ARG PATH="/opt/conda/bin:${PATH}"
WORKDIR /home

RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt update
RUN apt install -y htop wget tmux git python3-pip vim
RUN pip install pre-commit

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
RUN bash Miniconda3-py39_4.10.3-Linux-x86_64.sh -b -p /opt/conda
RUN rm -f Miniconda3-py39_4.10.3-Linux-x86_64.sh