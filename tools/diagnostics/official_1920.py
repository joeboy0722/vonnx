"""Unmodified official predictors: one predictor/thread/CUDA stream per model."""

import concurrent.futures
import queue
import threading
from types import SimpleNamespace
import numpy as np
from vonnx_tools.common import ROOT, manifest


class Official1920:
    def __init__(self, name):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.model = YOLO(str(ROOT / manifest(name)["weight"]))
        self.model.to("cuda")
        self.model.model.float()
        self.queue = queue.Queue(32)
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def submit_numpy(self, lane, frame, image, pixel_format):
        future = concurrent.futures.Future()
        self.queue.put_nowait((future, lane, frame, image.copy()))
        return future

    def work(self):
        stream = self.torch.cuda.Stream()
        while True:
            request = self.queue.get()
            if request is None:
                return
            future, lane, frame, image = request
            try:
                with self.torch.cuda.stream(stream), self.torch.inference_mode():
                    r = self.model.predict(
                        np.ascontiguousarray(image[:, :, ::-1]),
                        imgsz=1920,
                        rect=False,
                        half=False,
                        conf=0.25,
                        iou=0.7,
                        max_det=300,
                        quantize=None,
                        verbose=False,
                    )[0]
                    boxes = r.boxes.data.cpu().numpy()
                    keypoints = [] if r.keypoints is None else r.keypoints.data.cpu().numpy()
                    stream.synchronize()
                output = SimpleNamespace(
                    stream_id=lane,
                    frame_id=frame,
                    batch_size=1,
                    boxes=[
                        SimpleNamespace(x1=b[0], y1=b[1], x2=b[2], y2=b[3], score=b[4], class_id=int(b[5]))
                        for b in boxes
                    ],
                    keypoints=[
                        [SimpleNamespace(x=k[0], y=k[1], confidence=k[2]) for k in person]
                        for person in keypoints
                    ],
                    masks=[],
                )
                future.set_result([output])
            except Exception as e:
                future.set_exception(e)

    def close(self):
        self.queue.put(None)
        self.thread.join()
