import cv2
import sys
sys.path.insert(0, './app/object_detection')

from app.object_detection.detect import detect
from app.review_status.Status import Status


def review_status(filepath):
    cap = cv2.VideoCapture(filepath)
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    status = [Status.WORKING] * vid_len

    # cell phone detection
    status = detect(filepath, status)

    # sum up status
    status_sumup = [0] * len(Status)
    for s in status:
        status_sumup[s.value] += 1

    return status_sumup
    