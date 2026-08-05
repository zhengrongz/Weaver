import os
import sys
import time
import io
import tempfile
import numpy as np
import torch
import multiprocessing as mp
import matplotlib.pyplot as plt
import uvicorn
import asyncio
import concurrent.futures
import uuid
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
from tqdm import tqdm
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from asyncio import Semaphore
import re
import base64
import random
import json
import subprocess

# 必须在任何CUDA操作之前设置start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

PAD_IDX = -100

# 工具函数
def img_to_base64(img):
    image = Image.fromarray(img)
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)
    base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_string

async def bytes_to_numpy_video_pipe(file_contents: bytes) -> tuple[np.ndarray, dict]:
    """使用管道方式处理视频"""
    try:
        # 先获取视频信息
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', 
            '-i', 'pipe:'
        ]
        
        probe_process = subprocess.Popen(
            probe_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        probe_stdout, probe_stderr = probe_process.communicate(input=file_contents)
        
        if probe_process.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {probe_stderr.decode()}")
        
        probe_data = json.loads(probe_stdout.decode())
        video_stream = next((s for s in probe_data['streams'] if s['codec_type'] == 'video'), None)
        
        if not video_stream:
            raise ValueError("No video stream found")
        
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # 解码视频
        decode_cmd = [
            'ffmpeg', '-i', 'pipe:', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
        ]
        
        decode_process = subprocess.Popen(
            decode_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = decode_process.communicate(input=file_contents)
        
        if decode_process.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {stderr.decode()}")
        
        # 转换为 numpy 数组
        frames = np.frombuffer(stdout, dtype=np.uint8)
        frame_size = height * width * 3
        num_frames = len(frames) // frame_size
        video_array = frames.reshape(num_frames, height, width, 3)
        
        video_info = {
            'width': width,
            'height': height,
            'num_frames': num_frames
        }
        
        return video_array, video_info
        
    except Exception as e:
        raise RuntimeError(f"Video processing failed: {str(e)}")

def ask_inference(model, processor, video_dir, text_prompt, start, end):
    instruct = f"Does the query '{text_prompt}' happened in this video? Only Give answer of 'Yes' or 'No'. No need for reason."
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_dir,
                    "fps": 2.0,
                    "video_start": start,
                    "video_end": end
                },
                {"type": "text", "text": instruct},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    if output_text == "Yes":
        result = f"The query '{text_prompt}' happened in this video clip."
    else:
        result = f"The query '{text_prompt}' did not happen in this video clip."
    
    return result


def select_frame(model, processor, video, text_prompt):
    # instruct = f"Does this video match the query '{text_prompt}'? Only Give a confidence from 0 to 100 according to the degree of matching. No need for reason."
    instruct = f"""
            Evaluate how well the video matches the text prompt '{text_prompt}' based on the following criteria:

            1.  **Core Content**: Does the video feature the key subjects, objects, and scenes described in the prompt?
            2.  **Action & Plot**: Do the actions or events in the video align with the prompt's description?
            3.  **Style & Mood**: Does the video's overall feel (e.g., color tone, pacing, emotion) match the prompt's requirements?

            Based on a holistic assessment of these criteria, provide a matching score from 0 to 100. Output only the number, with no explanation.
            """
    batch_size = 10
    total_frames = video.shape[0]
    all_outputs = []
    
    for start_idx in range(0, total_frames, batch_size):
        end_idx = min(start_idx + batch_size, total_frames)
        batch_frames = video[start_idx:end_idx]
        
        messages_group = []
        for i in range(batch_frames.shape[0]):
            frame = batch_frames[i]
            frame_byte = img_to_base64(frame)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image;base64,{frame_byte}",
                        },
                        {"type": "text", "text": instruct},
                    ],
                }
            ]
            messages_group.append(messages)
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_group
        ]
        image_inputs, video_inputs = process_vision_info(messages_group)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        all_outputs.extend(output_text)

    int_list = [int(float(x)) for x in all_outputs]
    max_val = max(int_list)
    frame_idx = random.choice([i for i, val in enumerate(int_list) if val == max_val])
    # with open("/mnt/ali-sh-1/usr/shiyudi1/REVPT/tools/Qwen/Qwen_eval.log", "a") as f:
    #     print(f"{int_list}", file=f, flush=True)
    return frame_idx

def run_inference(model, processor, video_path, text_prompt, duration):
    num_segments = 10
    window_length = duration / num_segments
    windows = []
    
    for i in range(num_segments):
        start = i * window_length
        end = (i + 1) * window_length
        windows.append([start, end])
    
    instruct = f"Does the query '{text_prompt} happened or occured in this video'? Only Give an asnwer of 'Yes' or 'No'. No need for reason."
    select_windows = []
    for window in windows:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": 1.0, "max_frames": 128, "video_start": window[0], "video_end": window[1]},
                    {"type": "text", "text": instruct}
                ]
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if output_text == "Yes":
            select_windows.append(window)

    return select_windows

# Worker进程函数
def worker_process_func(model_path, device, worker_id, request_queue, result_queue, is_busy, request_count):
    """独立的worker进程函数"""
    try:
        print(f"Worker {worker_id}: Starting on {device}...")
        
        # 在子进程中重新导入
        import torch
        import os
        import sys
        import numpy as np
        from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        
        # 设置CUDA设备
        if 'cuda' in device:
            gpu_id = int(device.split(':')[1])
            torch.cuda.set_device(gpu_id)
            print(f"Worker {worker_id}: Set CUDA_VISIBLE_DEVICES={gpu_id}")
        
        print(f"Worker {worker_id}: Loading Qwen model to {device}...")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map={"": device}, 
            attn_implementation="flash_attention_2"
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_path)
        print(f"Worker {worker_id}: Model loaded to {device}")

        local_request_count = 0
        clean_interval = 10

        while True:
            try:
                try:
                    request_id, request_data = request_queue.get(timeout=1)
                except:
                    continue
                
                with is_busy.get_lock():
                    is_busy.value = 1
                
                print(f"Worker {worker_id}: Processing request {request_id}")
                
                inputs = request_data
                start_time = time.time()
                
                video_path = inputs.get("video_path", None)
                text_prompt = inputs.get("text_prompt", None)
                start = inputs.get("start", None)
                
                if video_path:
                    if not start:
                        duration = inputs.get("duration", None)
                        pred_windows = run_inference(model, processor, video_path, text_prompt, duration)
                        result = {
                            "pred_windows": pred_windows,
                            "worker_id": worker_id,
                            "device": device,
                            "process_time": time.time() - start_time
                        }
                    else:
                        end = inputs.get("end", None)
                        output = ask_inference(model, processor, video_path, text_prompt, start, end)
                        result = {
                            "output": output,
                            "worker_id": worker_id,
                            "device": device,
                            "process_time": time.time() - start_time
                        }
                else:
                    video = inputs.get("video", None)
                    frame_idx = select_frame(model, processor, video, text_prompt)
                    result = {
                        "frame_idx": frame_idx,
                        "worker_id": worker_id,
                        "device": device,
                        "process_time": time.time() - start_time
                    }
                    
                result_queue.put((request_id, result))
                print(f"Worker {worker_id}: Completed request {request_id} in {time.time() - start_time:.2f}s")
                
                local_request_count += 1
                with request_count.get_lock():
                    request_count.value += 1
                
                if local_request_count % clean_interval == 0:
                    if 'cuda' in device:
                        torch.cuda.empty_cache()
            
            except Exception as e:
                print(f"Worker {worker_id} processing error: {str(e)}")
                import traceback
                traceback.print_exc()
                try:
                    result_queue.put((request_id, {"error": str(e), "worker_id": worker_id}))
                except:
                    pass
            
            finally:
                with is_busy.get_lock():
                    is_busy.value = 0

    except Exception as e:
        print(f"Worker {worker_id} initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()


class QwenWorker:
    def __init__(self, model_path, device="cpu", worker_id=0):  
        self.device = device
        self.model_path = model_path
        self.worker_id = worker_id
        
        # 这些将在start方法中创建
        self.request_queue = None
        self.result_queue = None
        self.process = None
        self.is_busy = None
        self.request_count = None
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
    def start(self):
        """Start the model worker process"""
        # 在这里创建multiprocessing对象
        self.request_queue = mp.Queue(maxsize=100)
        self.result_queue = mp.Queue(maxsize=100)
        self.is_busy = mp.Value('i', 0)
        self.request_count = mp.Value('i', 0)
        
        self.process = mp.Process(
            target=worker_process_func,
            args=(
                self.model_path,
                self.device, 
                self.worker_id,
                self.request_queue, 
                self.result_queue,
                self.is_busy,
                self.request_count
            )
        )
        self.process.daemon = True
        self.process.start()
        print(f"Worker {self.worker_id} process started (PID: {self.process.pid})")

    async def process_request_async(self, inputs):
        """异步处理请求"""
        if not self.request_queue or not self.result_queue:
            raise RuntimeError("Worker not started")
            
        request_id = str(uuid.uuid4())
        
        def _sync_process():
            try:
                self.request_queue.put((request_id, inputs), timeout=10)
                
                start_time = time.time()
                timeout = 600
                
                while time.time() - start_time < timeout:
                    try:
                        result_id, result = self.result_queue.get(timeout=1)
                        if result_id == request_id:
                            return result
                    except:
                        continue
                
                raise TimeoutError(f"Request timeout after {timeout}s")
            
            except Exception as e:
                raise RuntimeError(f"Failed to process request: {str(e)}")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, _sync_process)
        return result

    def is_available(self):
        if not self.process or not self.is_busy:
            return False
        return (self.process.is_alive() and self.is_busy.value == 0)

    def get_queue_size(self):
        if not self.request_queue:
            return 0
        try:
            return self.request_queue.qsize()
        except:
            return 0

    def get_processed_count(self):
        if not self.request_count:
            return 0
        return self.request_count.value

    def stop(self):
        if self.executor:
            self.executor.shutdown(wait=True)
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.kill()


class WorkerManager:
    def __init__(self):
        self.workers: List[QwenWorker] = []
        self.current_index = 0
        
    def add_worker(self, worker: QwenWorker):
        self.workers.append(worker)
        
    def get_available_worker(self) -> Optional[QwenWorker]:
        """智能选择worker"""
        available_workers = [w for w in self.workers if w.is_available()]
        
        if available_workers:
            return min(available_workers, key=lambda w: w.get_queue_size())
        
        if self.workers:
            alive_workers = [w for w in self.workers if w.process and w.process.is_alive()]
            if alive_workers:
                return min(alive_workers, key=lambda w: w.get_queue_size())
                
        return None
    
    def get_worker_status(self):
        status = []
        for worker in self.workers:
            try:
                status.append({
                    "worker_id": worker.worker_id,
                    "device": worker.device,
                    "is_alive": worker.process.is_alive() if worker.process else False,
                    "is_busy": bool(worker.is_busy.value) if worker.is_busy else False,
                    "is_available": worker.is_available(),
                    "queue_size": worker.get_queue_size(),
                    "processed_count": worker.get_processed_count()
                })
            except Exception as e:
                status.append({
                    "worker_id": worker.worker_id,
                    "device": worker.device,
                    "is_alive": False,
                    "is_busy": False,
                    "is_available": False,
                    "queue_size": 0,
                    "processed_count": 0,
                    "error": str(e)
                })
        return status
    
    def get_load_balance_info(self):
        total_processed = sum(w.get_processed_count() for w in self.workers)
        total_queue_size = sum(w.get_queue_size() for w in self.workers)
        available_workers = len([w for w in self.workers if w.is_available()])
        
        return {
            "total_workers": len(self.workers),
            "available_workers": available_workers,
            "total_processed": total_processed,
            "total_queue_size": total_queue_size,
            "load_distribution": [w.get_processed_count() for w in self.workers]
        }


# 全局worker管理器
worker_manager = WorkerManager()
MAX_CONCURRENT_REQUESTS = 15
request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_manager
    
    # 自动检测GPU数量
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("No GPU devices available, will use CPU")
        gpu_count = 1
    else:
        print(f"Detected {gpu_count} GPU devices")
    
    default_config = {
        "model_path": os.environ.get(
            "QWEN_MODEL_PATH",
            "/mnt/ali-sh-1/dataset/zeus/shiyudi/models/Qwen2.5-VL-7B-Instruct",
        ),
    }
    
    # 为每个GPU创建worker
    workers_to_start = []
    for gpu_id in range(gpu_count):
        device = f"cuda:{gpu_id}" if gpu_count > 1 or torch.cuda.is_available() else "cpu"
        worker = QwenWorker(
            model_path=default_config["model_path"],
            device=device,
            worker_id=gpu_id
        )
        workers_to_start.append(worker)
        worker_manager.add_worker(worker)
    
    # 顺序启动workers
    print(f"Starting {len(workers_to_start)} workers...")
    for i, worker in enumerate(workers_to_start):
        print(f"Starting worker {i} on {worker.device}...")
        worker.start()
        time.sleep(3)  # 给每个worker启动时间
    
    # 等待所有workers就绪
    print("Waiting for all workers to be ready...")
    max_wait_time = 180
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        ready_workers = sum(1 for worker in workers_to_start 
                          if worker.process and worker.process.is_alive())
        print(f"Ready workers: {ready_workers}/{len(workers_to_start)}")
        
        if ready_workers == len(workers_to_start):
            break
        await asyncio.sleep(3)
    
    ready_workers = sum(1 for worker in workers_to_start 
                      if worker.process and worker.process.is_alive())
    
    print(f"Successfully initialized {ready_workers}/{len(workers_to_start)} workers")
    
    yield
    
    # Cleanup
    print("Shutting down all workers...")
    for worker in workers_to_start:
        worker.stop()


app = FastAPI(
    title="Qwen API",
    description="API for Temporal Counting with Multi-GPU Support",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/count")
async def temporal_counting(
    video_path: str = Form(...), 
    text_prompt: str = Form(...), 
    duration: float = Form(...)
):
    async with request_semaphore:
        try:
            worker = worker_manager.get_available_worker()
            if not worker:
                raise HTTPException(
                    status_code=503, 
                    detail="All workers are busy. Please try again later."
                )
            
            inputs = {
                "video_path": video_path,
                "text_prompt": text_prompt,
                "duration": duration
            }
            
            print(f"→ Count request assigned to Worker {worker.worker_id} ({worker.device}) [Queue: {worker.get_queue_size()}]")
            
            result = await worker.process_request_async(inputs)
            
            if "pred_windows" in result:
                return JSONResponse(content=result, status_code=200)
            
            if "error" in result:
                raise Exception(result["error"])
                
            return JSONResponse(content=result, status_code=200)
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/frame")
async def frame_selection(file: UploadFile = File(...), text_prompt: str = Form(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video")

    async with request_semaphore:
        try:
            worker = worker_manager.get_available_worker()
            if not worker:
                raise HTTPException(
                    status_code=503, 
                    detail="All workers are busy. Please try again later."
                )
            
            # Read uploaded video
            contents = await file.read()
            frames, video_info = await bytes_to_numpy_video_pipe(contents)
            
            inputs = {
                "video": frames,
                "text_prompt": text_prompt
            }
            
            print(f"→ Frame request assigned to Worker {worker.worker_id} ({worker.device}) [Queue: {worker.get_queue_size()}]")
            
            result = await worker.process_request_async(inputs)
            
            if "frame_idx" in result:
                return JSONResponse(content=result, status_code=200)
            
            if "error" in result:
                raise Exception(result["error"])
                
            return JSONResponse(content=result, status_code=200)
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/ask")
async def ask_happened(
    video_path: str = Form(...), 
    text_prompt: str = Form(...), 
    duration: float = Form(...),
    start: float = Form(...),
    end: float = Form(...)
):
    async with request_semaphore:
        try:
            worker = worker_manager.get_available_worker()
            if not worker:
                raise HTTPException(
                    status_code=503, 
                    detail="All workers are busy. Please try again later."
                )
            
            inputs = {
                "video_path": video_path,
                "text_prompt": text_prompt,
                "duration": duration,
                "start": start,
                "end": end
            }
            
            print(f"→ Count request assigned to Worker {worker.worker_id} ({worker.device}) [Queue: {worker.get_queue_size()}]")
            
            result = await worker.process_request_async(inputs)
            
            if "output" in result:
                return JSONResponse(content=result, status_code=200)
            
            if "error" in result:
                raise Exception(result["error"])
                
            return JSONResponse(content=result, status_code=200)
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/workers/status")
async def get_worker_status():
    return {
        "workers": worker_manager.get_worker_status(),
        "load_balance": worker_manager.get_load_balance_info()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    worker_status = worker_manager.get_worker_status()
    load_info = worker_manager.get_load_balance_info()
    
    total_workers = len(worker_status)
    active_workers = len([w for w in worker_status if w["is_alive"]])
    available_workers = len([w for w in worker_status if w["is_available"]])
    
    if active_workers == total_workers:
        status = "healthy"
    elif active_workers >= max(1, total_workers * 0.5):
        status = "degraded"
    else:
        status = "unhealthy"
    
    return {
        "status": status,
        "gpu_count": torch.cuda.device_count(),
        "active_workers": f"{active_workers}/{total_workers}",
        "available_workers": available_workers,
        "total_processed": load_info["total_processed"],
        "workers": worker_status
    }

