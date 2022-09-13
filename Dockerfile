# FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install python-setuptools -y
RUN easy_install pip

COPY requirements.txt /tmp/  
RUN pip install --upgrade pip  
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /app
COPY app/ /app/

WORKDIR /

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]