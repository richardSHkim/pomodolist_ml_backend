import time
import gc

import torch

from app.review_status.Status import Status
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import time_synchronized


def detect(source, status, model, device, half, stride, conf_thres=0.25, iou_thres=0.45):
    with torch.no_grad():
        imgsz = 640

        # Initialize
        set_logging()

        # Load model
        imgsz = check_img_size(imgsz, s=stride)  # check img_size


        # Set Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            if (dataset.frame % 10):
                continue
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=False)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0, frame = '', im0s, getattr(dataset, 'frame', 0)
                if frame >= len(status):
                    break

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        # check if cell phone exists
                        if names[int(c)] == 'cell phone':
                            status[frame] = Status.PHUBBING

                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    gc.collect()
    torch.cuda.empty_cache()

    print(f'Done. ({time.time() - t0:.3f}s)')
    return status
