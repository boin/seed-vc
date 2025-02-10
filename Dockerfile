FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

LABEL maintainer="ttd@server"
LABEL version="dev"
LABEL description="Docker image for seed-vc"


# Install 3rd party apps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata ffmpeg libsox-dev vim parallel aria2 git git-lfs locales && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

#RUN locale-gen zh_CN.UTF-8 

# Copy only requirements.txt initially to leverage Docker cache
WORKDIR /app

RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \   
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the rest of the application
COPY . /app

CMD [ "python", "api.py" ]