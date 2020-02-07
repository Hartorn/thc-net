FROM python:3.7-slim-buster@sha256:dc7110e9b6da8c75b703d17ff4d276ae65d7e3d21ab24847779cb442130710e1
RUN apt update && apt install curl make git libopenblas-base libgomp1 graphviz gcc g++ -y
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
ENV SHELL /bin/bash -l

ENV POETRY_CACHE /work/.cache/poetry
ENV PIP_CACHE_DIR /work/.cache/pip
ENV JUPYTER_RUNTIME_DIR /work/.cache/jupyter/runtime
ENV JUPYTER_CONFIG_DIR /work/.cache/jupyter/config

RUN $HOME/.poetry/bin/poetry config virtualenvs.path $POETRY_CACHE

ENV PATH /root/.poetry/bin:/bin:/usr/local/bin:/usr/bin

CMD ["bash", "-l"]