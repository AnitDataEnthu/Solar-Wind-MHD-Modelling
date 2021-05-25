# install base image
FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    libjpeg8-dev

RUN ln -s /usr/lib/x86_64-linux-gnu/libjpeg.so /usr/lib
RUN ln -s /usr/lib/x86_64-linux-gnu/libfreetype.so /usr/lib
RUN ln -s /usr/lib/x86_64-linux-gnu/libz.so /usr/lib

RUN apt-get update &&\
    apt-get install python3.7 -y &&\
    apt-get install python3-pip -y &&\
    apt-get install graphviz -y

# exposing default port for streamlit
EXPOSE 8501

WORKDIR /root

RUN mkdir /notebooks

WORKDIR /home/agupta33/present

# cmd to launch app when container is run
CMD streamlit run mainpage.py

# running and configuring jupyter
RUN jupyter notebook --generate-config

ENV CONFIG_PATH="/root/.jupyter/jupyter_notebook_config.py"


COPY "jupyter_notebook_config.py" ${CONFIG_PATH}

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

ENTRYPOINT ["sh", "-c", "jupyter notebook --allow-root -y --no-browser --ip=0.0.0.0 --config=${CONFIG_PATH}"]

