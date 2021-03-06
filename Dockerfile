FROM bazire/python:3.7-cpu

RUN apt update && apt install curl make git -y
RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
SHELL ["/bin/bash", "-lc"]

ENV POETRY_CACHE /work/.cache/poetry
ENV PIP_CACHE_DIR /work/.cache/pip
ENV JUPYTER_RUNTIME_DIR /work/.cache/jupyter/runtime
ENV JUPYTER_CONFIG_DIR /work/.cache/jupyter/config

RUN $HOME/.poetry/bin/poetry config virtualenvs.path $POETRY_CACHE

ENV PATH ${PATH}:/root/.poetry/bin:/bin:/usr/local/bin:/usr/bin:/usr/sbin:/usr/local/sbin:/sbin

# Install Go (for TF build)
RUN curl https://dl.google.com/go/go1.13.10.linux-amd64.tar.gz -o go.tar.gz \
&& tar -xvf go.tar.gz \
&& rm -rf go.tar.gz \
&& mv go /usr/local

ENV GOROOT /usr/local/go
ENV PATH $PATH:$GOPATH/bin:$GOROOT/bin

# Install Bazelisk as Bazel
RUN go get github.com/bazelbuild/bazelisk && ln -s /root/go/bin/bazelisk /usr/local/bin/bazel

CMD ["bash", "-l"]