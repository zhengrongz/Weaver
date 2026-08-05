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
from asyncio import Semaphore
import cv2
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import copy
import json
import subprocess

# 必须在任何CUDA操作之前设置start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

# 工具函数
def find_boundaries_torch(mask):
    from skimage.segmentation import find_boundaries
    image_data = np.where(mask, 255, 0).astype(np.uint8)
    mask_np = mask.to(torch.bool).numpy()
    boundaries = find_boundaries(mask_np, mode='outer')
    boundary_points = np.argwhere(boundaries)
    if boundary_points.size == 0:
        return torch.tensor([-1, -1, -1, -1], dtype=torch.bfloat16)
    h0, w0 = boundary_points.min(axis=0)
    h1, w1 = boundary_points.max(axis=0)
    return torch.tensor([w0 / mask.shape[1], h0 / mask.shape[0], w1 / mask.shape[1], h1 / mask.shape[0]], dtype=torch.bfloat16)

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

# Worker进程函数
# Worker进程函数 - 修复版本
def worker_process_func(grounding_dino_model_id, device, worker_id, request_queue, result_queue, is_busy, request_count):
    """独立的worker进程函数"""
    try:
        print(f"Worker {worker_id}: Starting on {device}...")
        
        # 在子进程中重新导入
        import torch
        import os
        import sys
        import numpy as np
        import copy
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        
        # 重新导入SAM2相关模块
        sys.path.append(os.environ.get('GROUNDEDSAM2_TOOL_DIR', '/mnt/ali-sh-1/usr/shiyudi1/REVPT/tools/GroundedSAM2'))
        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor 
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from utils.track_utils import sample_points_from_masks
        from utils.video_utils import create_video_from_images
        from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
        

        if 'cuda' in device:
            gpu_id = int(device.split(':')[1])
            torch.cuda.set_device(gpu_id)
            print(f"Worker {worker_id}: Set CUDA_VISIBLE_DEVICES={gpu_id}")
        
        # SAM2配置
        sam2_checkpoint = os.environ.get(
            'SAM2_CHECKPOINT',
            '/mnt/ali-sh-1/dataset/zeus/shiyudi/models/sam2/sam2.1_hiera_large.pt',
        )
        model_cfg = os.environ.get(
            'SAM2_MODEL_CFG',
            '/mnt/ali-sh-1/usr/shiyudi1/REVPT/tools/GroundedSAM2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
        )
        
        print(f"Worker {worker_id}: Loading GroundedSAM2 model to {device}...")
        
        # 设置CUDA优化
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 加载模型
        video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        image_predictor = SAM2ImagePredictor(sam2_image_model)
        
        processor = AutoProcessor.from_pretrained(grounding_dino_model_id)
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_dino_model_id).to(device)
        
        # 图像变换
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean, std)
        type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
        transform = transforms.Compose([   
            type_transform,
            transforms.Resize((1024, 1024), interpolation=InterpolationMode.BICUBIC),
            normalize,
        ])
        
        print(f"Worker {worker_id}: Model loaded to {device}")
        
        local_request_count = 0
        clean_interval = 5

        while True:
            try:
                try:
                    request_id, request_data = request_queue.get(timeout=1)
                except:
                    continue
                
                with is_busy.get_lock():
                    is_busy.value = 1
                
                print(f"Worker {worker_id}: Processing request {request_id}")
                
                video_numpy, objects = request_data
                start_time = time.time()
                
                height, width = video_numpy.shape[1:3]
                video = torch.tensor(video_numpy, device=device)
                transformed_video = transform(video.permute(0,3,1,2))
                transformed_video = transformed_video.to(torch.bfloat16) 
                transformed_video = transformed_video.unsqueeze(0)

                inference_state = video_predictor.init_state_images(images=transformed_video, video_height=height, video_width=width)
                total_bboxes = []

                sam2_masks = MaskDictionaryModel()
                PROMPT_TYPE_FOR_VIDEO = "mask"
                objects_count = 0
                obj_nums = {}
                for obj in objects:
                    if not obj.endswith('.'):
                        obj = obj + "."
                    obj = obj.lower()
                    
                    total_video_segments = []
                    for frame_idx in range(0, video_numpy.shape[0], 4):
                        ann_image = Image.fromarray(video_numpy[frame_idx])
                        mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{obj}.npy")

                        inputs = processor(images=ann_image, text=obj, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = grounding_model(**inputs)

                        results = processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            box_threshold=0.4,
                            text_threshold=0.3,
                            target_sizes=[ann_image.size[::-1]]
                        )

                        image_predictor.set_image(np.array(ann_image.convert("RGB")))
                        input_boxes = results[0]["boxes"]
                        obj_num = input_boxes.shape[0]
                        if obj not in obj_nums:
                            obj_nums[obj] = obj_num
                        else:
                            obj_nums[obj] = max(obj_num, obj_nums[obj])

                        OBJECTS = results[0]["labels"]
                        
                        if input_boxes.shape[0] != 0:
                            masks, scores, logits = image_predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=input_boxes,
                                multimask_output=False,
                            )
                            
                            if masks.ndim == 2:
                                masks = masks[None]
                                scores = scores[None]
                                logits = logits[None]
                            elif masks.ndim == 4:
                                masks = masks.squeeze(1)
                            
                            if mask_dict.promote_type == "mask":
                                mask_dict.add_new_frame_annotation(
                                    mask_list=torch.tensor(masks).to(device), 
                                    box_list=torch.tensor(input_boxes), 
                                    label_list=OBJECTS
                                )
                            else:
                                raise NotImplementedError("SAM 2 video predictor only support mask prompts")

                            objects_count = mask_dict.update_masks(
                                tracking_annotation_dict=sam2_masks, 
                                iou_threshold=0.8, 
                                objects_count=objects_count
                            )
                        else:
                            mask_dict = sam2_masks

                        if len(mask_dict.labels) == 0:
                            continue
                        else: 
                            video_predictor.reset_state(inference_state)

                            for object_id, object_info in mask_dict.labels.items():
                                frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                                    inference_state,
                                    frame_idx,
                                    object_id,
                                    object_info.mask,
                                )
        
                            video_segments = {}
                            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                                inference_state, max_frame_num_to_track=3, start_frame_idx=frame_idx
                            ):
                                frame_masks = MaskDictionaryModel()
                                
                                for i, out_obj_id in enumerate(out_obj_ids):
                                    out_mask = (out_mask_logits[i] > 0.0)
                                    object_info = ObjectInfo(
                                        instance_id=out_obj_id, 
                                        mask=out_mask[0], 
                                        class_name=mask_dict.get_target_class_name(out_obj_id)
                                    )
                                    object_info.update_box()
                                    frame_masks.labels[out_obj_id] = object_info
                                    frame_masks.mask_name = f"mask_{obj}.npy"
                                    frame_masks.mask_height = out_mask.shape[-2]
                                    frame_masks.mask_width = out_mask.shape[-1]

                                video_segments[out_frame_idx] = frame_masks
                                sam2_masks = copy.deepcopy(frame_masks)
                            
                            total_video_segments.append(video_segments)
                        
                    bboxes = {}
                    for video_segment in total_video_segments:
                        key_list = list(video_segment.keys())
                        for key in key_list:
                            labels = video_segment[key].labels
                            label_key_list = list(labels.keys())
                            for label_key in label_key_list:
                                if label_key not in bboxes:
                                    bboxes[label_key] = np.zeros((video_numpy.shape[0], 4))
                                label = labels[label_key]
                                bbox = np.array([label.x1, label.y1, label.x2, label.y2])
                                bboxes[label_key][key] = bbox

                    bbox_key_list = list(bboxes.keys())
                    for key in bbox_key_list:
                        bboxes[key] = bboxes[key].tolist()

                    total_bboxes.append(bboxes)

                process_time = time.time() - start_time
                result = {
                    "pred_bboxes": total_bboxes,
                    "obj_nums": obj_nums,
                    "worker_id": worker_id,
                    "device": device,
                    "process_time": process_time
                }
                    
                result_queue.put((request_id, result))
                print(f"Worker {worker_id}: Completed request {request_id} in {process_time:.2f}s")
                
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


class GroundedSAM2Worker:
    def __init__(self, grounding_dino_model_id, device="cpu", worker_id=0):  
        self.device = device
        self.model_id = grounding_dino_model_id
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
        self.request_queue = mp.Queue(maxsize=50)
        self.result_queue = mp.Queue(maxsize=50)
        self.is_busy = mp.Value('i', 0)
        self.request_count = mp.Value('i', 0)
        
        self.process = mp.Process(
            target=worker_process_func,
            args=(
                self.model_id,
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

    async def process_request_async(self, video, text_prompt):
        """异步处理请求"""
        if not self.request_queue or not self.result_queue:
            raise RuntimeError("Worker not started")
            
        request_id = str(uuid.uuid4())
        
        def _sync_process():
            try:
                self.request_queue.put((request_id, (video, text_prompt)), timeout=15)
                
                start_time = time.time()
                timeout = 240  # 4分钟超时
                
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
            self.process.join(timeout=15)
            if self.process.is_alive():
                self.process.kill()


class WorkerManager:
    def __init__(self):
        self.workers: List[GroundedSAM2Worker] = []
        self.current_index = 0
        
    def add_worker(self, worker: GroundedSAM2Worker):
        self.workers.append(worker)
        
    def get_available_worker(self) -> Optional[GroundedSAM2Worker]:
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
MAX_CONCURRENT_REQUESTS = 10  # SAM2比较消耗资源，限制并发数
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
        "grounding_dino_model_id": os.environ.get(
            "GROUNDING_DINO_MODEL_ID",
            "/mnt/ali-sh-1/dataset/zeus/shiyudi/models/grounding-dino-base",
        ),
    }
    
    # 为每个GPU创建worker
    workers_to_start = []
    for gpu_id in range(gpu_count):
        device = f"cuda:{gpu_id}" if gpu_count > 1 or torch.cuda.is_available() else "cpu"
        worker = GroundedSAM2Worker(
            grounding_dino_model_id=default_config["grounding_dino_model_id"],
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
        time.sleep(5)  # SAM2启动时间较长，给更多时间
    
    # 等待所有workers就绪
    print("Waiting for all workers to be ready...")
    max_wait_time = 300  # 5分钟超时
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        ready_workers = sum(1 for worker in workers_to_start 
                          if worker.process and worker.process.is_alive())
        print(f"Ready workers: {ready_workers}/{len(workers_to_start)}")
        
        if ready_workers == len(workers_to_start):
            break
        await asyncio.sleep(5)
    
    ready_workers = sum(1 for worker in workers_to_start 
                      if worker.process and worker.process.is_alive())
    
    print(f"Successfully initialized {ready_workers}/{len(workers_to_start)} workers")
    
    yield
    
    # Cleanup
    print("Shutting down all workers...")
    for worker in workers_to_start:
        worker.stop()


app = FastAPI(
    title="GroundedSAM2 API",
    description="API for GroundedSAM2 Tracking with Multi-GPU Support",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/tracking")
async def tracking_objects(file: UploadFile = File(...), text_prompt: List[str] = Form(...)):
    async with request_semaphore:
        try:
            worker = worker_manager.get_available_worker()
            if not worker:
                raise HTTPException(
                    status_code=503, 
                    detail="All workers are busy. Please try again later."
                )
            
            # Read uploaded file
            contents = await file.read()
            if file.content_type.startswith("video/"):
                frames, video_info = await bytes_to_numpy_video_pipe(contents)
            else:
                image_pil = Image.open(io.BytesIO(contents))
                if image_pil.mode != "RGB":
                    image_pil = image_pil.convert("RGB")
                image_np = np.array(image_pil)
                frames = np.expand_dims(image_np, axis=0)
            
            print(f"→ Tracking request assigned to Worker {worker.worker_id} ({worker.device}) [Queue: {worker.get_queue_size()}]")
            
            result = await worker.process_request_async(frames, text_prompt)
            
            if "pred_bboxes" in result:
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
