import os

from google.cloud import storage
import ffmpeg
from tempfile import NamedTemporaryFile
import torch
from fastapi import FastAPI, File, UploadFile

from app.object_detection.utils.torch_utils import select_device
from app.object_detection.models.experimental import attempt_load
from app.review_status.review_status import review_status


torch.multiprocessing.set_start_method('spawn')


WEIGHT_DIR = f'./app/object_detection/weights'
BUCKET_NAME = 'pomodolist-yolo'
BLOB_NAME = 'yolov7.pt'
if not os.path.exists(os.path.join(WEIGHT_DIR, BLOB_NAME)):
    storage_client = storage.Client()
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(BLOB_NAME)
    blob.download_to_filename(os.path.join(WEIGHT_DIR, BLOB_NAME))


def create_app():
    # load detection model
    weights = os.path.join(WEIGHT_DIR, BLOB_NAME)
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(weights, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16
    stride = int(model.stride.max())

    app = FastAPI()
    return app, model, device, half, stride


app, model, device, half, stride = create_app()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/pyapi/review")
def review(video: UploadFile = File(...)):
    # copy data to temporary file
    with NamedTemporaryFile("wb", delete=False) as temp:
        try:
            contents = video.file.read()
            temp.write(contents)
        except Exception as e:
            print(e)
            return {"message": "There was an error uploading the file"}
        finally:
            video.close()
    
    # encoding with ffmpeg
    with NamedTemporaryFile("wb", delete=False) as encoded_temp:
        try:
            stream = ffmpeg.input(temp.name)
            stream = ffmpeg.output(stream, encoded_temp.name+'.mp4')
            ffmpeg.run(stream)
        except Exception as e:
            print(e)
            return {"message": "There was an error encoding the file"}
        finally:
            os.remove(temp.name)

    # review status
    try:
        status = review_status(encoded_temp.name+'.mp4', model, device, half, stride)
    except Exception as e:
        print(e)
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(encoded_temp.name+'.mp4')

    return {"status": status}
