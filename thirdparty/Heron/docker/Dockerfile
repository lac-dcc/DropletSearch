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

RUN pip3 install numpy==1.20 decorator scipy pytest psutil typing_extensions tornado cloudpickle \ 
    'xgboost<1.6.0' synr mxnet

RUN curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash

RUN echo "export PYTHON_BUILD_ARIA2_OPTS='-x 10 -k 1M'" >> ~/.bashrc
RUN echo "export PATH='~/.pyenv/bin:$PATH'" >> ~/.bashrc
RUN echo "eval '$(pyenv init --path)'" >> ~/.bashrc
RUN echo "eval '$(pyenv init -)'" >> ~/.bashrc
RUN echo "eval '$(pyenv virtualenv-init -)'" >> ~/.bashrc

########################################### COPY HAMSA ##################################################
RUN cd $HOME && git clone https://github.com/lac-dcc/garkain 

RUN sh $HOME/garkain/thirdparty/Heron/docker/update_bash.sh

RUN pyenv install 3.6.10
RUN pyenv global 3.6.10

########################################### INSTALL TVM #################################################
RUN cd $HOME && git clone -b v0.10.0 --recursive https://github.com/apache/tvm && \
    cd tvm && mkdir -p build && \
    cp $HOME/garkain/docker/cuda/config.cmake $HOME/tvm/build && \
    cd build && cmake .. && make -j8 && \
    cd .. && sudo make cython3

RUN echo "export PYTHONPATH=/root/tvm/python:/python:" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-12.0/compat/" >> ~/.bashrc

RUN cd $HOME/garkain/thirdparty/Heron && pip3.6 install -r requirements.txt

########################################### WORKDIR #####################################################
WORKDIR /root/garkain/thirdparty/Heron
ENTRYPOINT ["/bin/bash"]
