import cv2

from app.object_detection.detect import detect
from app.review_status.Status import Status


def review_status(filepath, model, device, half, stride):
    cap = cv2.VideoCapture(filepath)
    vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    status = [Status.WORKING] * vid_len

    # cell phone detection
    status = detect(filepath, status, model, device, half, stride)

    # sum up status
    status_sumup = [0] * len(Status)
    for s in status:
        status_sumup[s.value] += 1

    return status_sumup
    