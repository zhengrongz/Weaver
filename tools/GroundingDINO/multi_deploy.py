"""
Grounding DINO spatial grounding service.
Performs per-frame object detection on video/image inputs.
Returns bounding boxes in the same format as the GroundedSAM2 tracking service,
so SpatialGroundingTool can switch between the two without code changes.

API:
  POST /grounding
    - file: video (mp4) or image (jpeg/png)
    - text_prompt: JSON-encoded list of object names, e.g. '["dog", "cat"]'
  GET  /health

Launch via lanuch_tools.py (see train_tools_config.json / eval_tools_config.json).
Can also be started directly:
  python multi_deploy.py
or via uvicorn:
  uvicorn multi_deploy:app --host 0.0.0.0 --port 9998
"""

import os
import sys
import io
import json
import time
import subprocess
import numpy as np
import torch
import uvicorn
import asyncio
import concurrent.futures

from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List, Optional
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ──────────────────────────────────────────────
# Config (all overridable via environment variables)
# ──────────────────────────────────────────────
MODEL_ID = os.environ.get(
    "GROUNDING_DINO_MODEL_ID",
    "path/to/models/grounding-dino-base",
)
BOX_THRESHOLD = float(os.environ.get("BOX_THRESHOLD", "0.35"))
TEXT_THRESHOLD = float(os.environ.get("TEXT_THRESHOLD", "0.25"))
PORT = int(os.environ.get("GROUNDING_DINO_PORT", "9998"))
# Sample every N frames for video inputs (1 = every frame)
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "1"))

# ──────────────────────────────────────────────
# Global model (loaded once at startup)
# ──────────────────────────────────────────────
_processor: Optional[AutoProcessor] = None
_model = None
_device: str = "cpu"
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="gdino")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _load_model():
    global _processor, _model, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GroundingDINO] Loading model from {MODEL_ID} on {_device} ...")
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(_device)
    _model.eval()
    print(f"[GroundingDINO] Model loaded successfully.")


async def _bytes_to_frames(contents: bytes, content_type: str) -> np.ndarray:
    """
    Decode uploaded bytes to a numpy array of shape (T, H, W, 3) uint8.
    For images, T=1.
    """
    if content_type.startswith("video/"):
        # Probe video dimensions
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-i", "pipe:"
        ]
        probe_proc = subprocess.Popen(
            probe_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        probe_out, probe_err = probe_proc.communicate(input=contents)
        if probe_proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {probe_err.decode()}")

        probe_data = json.loads(probe_out.decode())
        video_stream = next(
            (s for s in probe_data["streams"] if s["codec_type"] == "video"), None
        )
        if not video_stream:
            raise ValueError("No video stream found in uploaded file.")

        width = int(video_stream["width"])
        height = int(video_stream["height"])

        # Decode all frames to raw RGB
        decode_cmd = ["ffmpeg", "-i", "pipe:", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        decode_proc = subprocess.Popen(
            decode_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = decode_proc.communicate(input=contents)
        if decode_proc.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {stderr.decode()}")

        frames = np.frombuffer(stdout, dtype=np.uint8)
        frame_size = height * width * 3
        num_frames = len(frames) // frame_size
        video_array = frames.reshape(num_frames, height, width, 3)
        return video_array
    else:
        # Single image → wrap as 1-frame "video"
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        return np.expand_dims(np.array(pil_img), axis=0)  # (1, H, W, 3)


def _detect_objects_on_frame(
    frame_rgb: np.ndarray,
    objects: List[str],
    h: int,
    w: int,
) -> dict:
    """
    Run Grounding DINO on a single RGB frame (H, W, 3) uint8.
    For each object, returns the best-scoring box as [x1, y1, x2, y2] (absolute pixels).
    Returns [0, 0, 0, 0] if no box is found.
    """
    pil_img = Image.fromarray(frame_rgb.astype(np.uint8))
    frame_boxes = {}

    for obj in objects:
        text = obj.lower().strip()
        if not text.endswith("."):
            text = text + "."

        inputs = _processor(images=pil_img, text=text, return_tensors="pt").to(_device)
        with torch.no_grad():
            outputs = _model(**inputs)

        results = _processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(h, w)],
        )

        boxes = results[0]["boxes"]   # (N, 4) xyxy absolute coords
        scores = results[0]["scores"]

        if boxes.shape[0] > 0:
            best_idx = scores.argmax().item()
            x1, y1, x2, y2 = boxes[best_idx].tolist()
            frame_boxes[obj] = [x1, y1, x2, y2]
        else:
            frame_boxes[obj] = [0, 0, 0, 0]

    return frame_boxes


def _run_grounding(frames: np.ndarray, objects: List[str]) -> dict:
    """
    Iterate over sampled frames and run Grounding DINO per frame per object.

    Output format mirrors GroundedSAM2 /tracking so that highlight_video()
    in qwen_tools.py works without modification:

      pred_bboxes: List[Dict[int, List[List[float]]]]
        One dict per object. Key = object_idx (int), value = (T, 4) bbox list.
      obj_nums: Dict[str, int]
        Object name -> 1 if detected in any frame, else 0.
    """
    T, H, W, _ = frames.shape
    sampled_indices = list(range(0, T, FRAME_STRIDE))

    total_bboxes = []
    obj_nums = {}

    for obj_idx, obj in enumerate(objects):
        bbox_array = np.zeros((T, 4), dtype=np.float32)  # default [0,0,0,0]
        detected = False

        # Per-frame detection
        for frame_idx in sampled_indices:
            frame = frames[frame_idx]
            frame_boxes = _detect_objects_on_frame(frame, [obj], H, W)
            box = frame_boxes[obj]
            bbox_array[frame_idx] = box
            if box != [0, 0, 0, 0]:
                detected = True

        # Fill non-sampled frames by nearest-neighbor interpolation
        if FRAME_STRIDE > 1:
            sampled_set = set(sampled_indices)
            for t in range(T):
                if t not in sampled_set:
                    nearest = min(sampled_indices, key=lambda s: abs(s - t))
                    bbox_array[t] = bbox_array[nearest]

        obj_nums[obj] = 1 if detected else 0
        # Match GroundedSAM2 format: {object_id: [[x1,y1,x2,y2], ...] per frame}
        total_bboxes.append({obj_idx: bbox_array.tolist()})

    return {"pred_bboxes": total_bboxes, "obj_nums": obj_nums}


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="Grounding DINO Spatial Grounding API",
    description="Per-frame object detection using Grounding DINO (no SAM2 / no tracking).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/grounding")
async def grounding_endpoint(
    file: UploadFile = File(...),
    text_prompt: str = Form(...),   # JSON-encoded list, e.g. '["dog","cat"]'
):
    """
    Detect objects in each frame of the uploaded video/image.

    - **file**: video (mp4) or image (jpeg/png)
    - **text_prompt**: JSON string of object names, e.g. `'["person", "car"]'`
    """
    try:
        objects: List[str] = json.loads(text_prompt)
        if not isinstance(objects, list):
            raise ValueError("text_prompt must be a JSON list of strings.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid text_prompt: {e}")

    try:
        contents = await file.read()
        frames = await _bytes_to_frames(contents, file.content_type or "image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode file: {e}")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, _run_grounding, frames, objects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grounding failed: {e}")

    return JSONResponse(content=result, status_code=200)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_id": MODEL_ID,
        "device": _device,
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "frame_stride": FRAME_STRIDE,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
