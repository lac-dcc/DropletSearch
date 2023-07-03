################################################# Image #################################################
FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

############################################# Date and Time #############################################
ENV TZ="America/Sao_Paulo"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/* 

RUN apt-get update -y && apt-get install -y locales curl wget tar sudo git apt-utils \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG en_US.utf8

############################################# DEPENDENCIES ##############################################
RUN apt-get install -y gcc g++ graphviz vim python3 python3-pip python3-dev automake make \
    build-essential cmake llvm-8 clang cython3 libedit-dev libxml2-dev libtinfo-dev \
    python3-setuptools build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \ 
    gettext libncurses5-dev tk-dev tcl-dev blt-dev libgdbm-dev git python3-dev aria2 lzma-dev

RUN pip3 install ortools==9.1.9490 matplotlib==3.3.1 numpy==1.19.5 cloudpickle==1.2.2 decorator scipy \ 
    pytest psutil typing_extensions tornado synr mxnet protobuf==3.20.3 'xgboost<1.6.0'

########################################### COPY REPO ##################################################
RUN cd $HOME && git clone https://github.com/lac-dcc/garkain     

########################################### INSTALL TVM #################################################
RUN cd $HOME && git clone -b v0.8.0 --recursive https://github.com/apache/tvm && \
    cd tvm && mkdir -p build && \
    cp $HOME/garkain/docker/cuda/config.cmake $HOME/tvm/build && \
    cd build && cmake .. && make -j8 && \
    cd .. && sudo make cython3

RUN echo "export PYTHONPATH=/root/tvm/python:/root/garkain/thirdparty" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-12.0/compat/" >> ~/.bashrc

########################################### WORKDIR #####################################################
WORKDIR /root/garkain/thirdparty/Heron
ENTRYPOINT ["/bin/bash"]
