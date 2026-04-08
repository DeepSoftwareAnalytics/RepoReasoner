FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    tree \
    curl \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

#
# RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple/
# 
# RUN pip config set global.trusted-host repo.huaweicloud.com

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt pytest hunter

COPY runnable_agent_batch/ ./runnable_agent_batch/

CMD ["/bin/bash"]