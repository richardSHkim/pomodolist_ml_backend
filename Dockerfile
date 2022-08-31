FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

COPY . /workspace/pomodoro_fastapi/
WORKDIR /workspace/pomodoro_fastapi/

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app"]