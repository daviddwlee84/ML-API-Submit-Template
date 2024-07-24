FROM python:3.10-slim

# Create a new sources list file with Tsinghua mirror
# https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list

RUN apt-get update && apt-get install -y curl

# https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mlflow[extras] psycopg2-binary boto3 cryptography pymysql

EXPOSE 5000