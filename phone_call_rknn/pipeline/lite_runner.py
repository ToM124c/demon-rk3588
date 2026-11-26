import time

from rknnlite.api import RKNNLite

from ..config import resolve_rknn_path
from ..core.preprocess import preprocess


class RKNNRunner:
    def __init__(self, model_path, imgsz, input_layout):
        self.model_path = resolve_rknn_path(model_path)
        self.imgsz = imgsz
        self.input_layout = input_layout
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {self.model_path}, ret={ret}")
        ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN runtime: {self.model_path}, ret={ret}")

    def infer(self, frame):
        runner_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        tensor, scale, pad_x, pad_y = preprocess(frame, self.imgsz, self.input_layout)
        preprocess_sec = time.perf_counter() - preprocess_start

        inference_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[tensor])
        rknn_inference_sec = time.perf_counter() - inference_start
        if outputs is None:
            raise RuntimeError(f"RKNN inference returned None: {self.model_path}")
        total_runner_sec = time.perf_counter() - runner_start
        timing = {
            "preprocess_ms": preprocess_sec * 1000.0,
            "postprocess_ms": 0.0,
            "rknn_inference_ms": rknn_inference_sec * 1000.0,
            "total_runner_ms": total_runner_sec * 1000.0,
        }
        return outputs, scale, pad_x, pad_y, timing

    def infer_tensor(self, tensor):
        inference_start = time.perf_counter()
        outputs = self.rknn.inference(inputs=[tensor])
        rknn_inference_sec = time.perf_counter() - inference_start
        if outputs is None:
            raise RuntimeError(f"RKNN inference returned None: {self.model_path}")
        timing = {
            "preprocess_ms": 0.0,
            "postprocess_ms": 0.0,
            "rknn_inference_ms": rknn_inference_sec * 1000.0,
            "total_runner_ms": rknn_inference_sec * 1000.0,
        }
        return outputs, timing

    def release(self):
        self.rknn.release()

