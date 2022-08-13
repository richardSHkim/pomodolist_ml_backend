import os
import ffmpeg
import aiofiles
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, UploadFile


from app.review_status.review_status import review_status


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/pyapi/review")
async def review(video: UploadFile = File(...)):
    # copy data to temporary file
    async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
        try:
            contents = await video.read()
            await temp.write(contents)
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            await video.close()
    
    # encoding with ffmpeg
    with NamedTemporaryFile("wb", delete=False) as encoded_temp:
        try:
            stream = ffmpeg.input(temp.name)
            stream = ffmpeg.output(stream, encoded_temp.name+'.mp4')
            ffmpeg.run(stream)
        except Exception as err:
            print(err)
            return {"message": err}
        finally:
            os.remove(temp.name)

    # review status
    try:
        status = review_status(encoded_temp.name+'.mp4')
    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(encoded_temp.name+'.mp4')

    return {"status": status}
