FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /tmp/  
RUN pip install --upgrade pip  
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /app
COPY app/ /app/

WORKDIR /

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]