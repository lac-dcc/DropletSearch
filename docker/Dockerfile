################################################# Image #################################################
FROM ubuntu:22.04

############################################# Date and Time #############################################
ENV TZ="America/Sao_Paulo"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/* 

RUN apt-get update -y && apt-get install -y locales curl wget tar sudo git apt-utils \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8

############################################# DEPENDENCIES ##############################################
RUN apt-get install -y gcc g++ graphviz vim python3 python3-pip python3-dev automake make \
    build-essential cmake llvm-dev clang cython3 python-is-python3

########################################### COPY HAMSA ##################################################
RUN cd $HOME && git clone https://github.com/lac-dcc/garkain 

########################################### INSTALL TVM #################################################
RUN cd $HOME && git clone -b v0.10.0 --recursive https://github.com/apache/tvm && \
    cd tvm && mkdir -p build && \
    cp $HOME/garkain/docker/utils/config.cmake $HOME/tvm/build && \
    cp $HOME/garkain/docker/utils/droplet_algorithm/*.py $HOME/tvm/python/tvm/autotvm/tuner/ && \ 
    cd build && cmake .. && make -j4 \
    echo "export TVM_HOME=/home/root/tvm" >> ~/.bashrc && \
    echo "export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}" >> ~/.bashrc \
    cd .. && sudo make cython3

########################################### WORKDIR #####################################################
WORKDIR /root/garkain
ENTRYPOINT ["/bin/bash"]
