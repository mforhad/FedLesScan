FROM public.ecr.aws/lambda/python:3.10

ENV TF_ENABLE_ONEDNN_OPTS=1

# Install tensorflow beforehand to improve docker build/push times
RUN pip install tensorflow

RUN yum install -y git

# Install fedless from built wheel
COPY ./dist/fedless*.whl .
RUN pip install fedless*.whl