FROM cnstark/pytorch:2.0.1-py3.10.11-cuda11.8.0-ubuntu22.04

LABEL maintainer="ttd@server"
LABEL version="dev"
LABEL description="Docker image for seed-vc"


# Install 3rd party apps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata ffmpeg libsox-dev vim parallel aria2 git git-lfs locales && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

#RUN locale-gen zh_CN.UTF-8 

# Copy only requirements.txt initially to leverage Docker cache
WORKDIR /app

COPY requirements.txt /
# COPY requirements_web_demo.txt /workspace/
RUN pip install --no-cache-dir -r /requirements.txt
#RUN pip install --no-cache-dir -r requirements_web_demo.txt

# Copy the rest of the application
COPY . /app

CMD [ "python", "app.py" ]