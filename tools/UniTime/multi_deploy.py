import os
import sys
import time
import numpy as np
import torch
import multiprocessing as mp
import uvicorn
import asyncio
import concurrent.futures
import uuid
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, List
from contextlib import asynccontextmanager
from asyncio import Semaphore
import re

# 必须在任何CUDA操作之前设置start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过了

# Import your models
from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor
from qwen_vision_process import process_vision_info
from feature import feature

PAD_IDX = -100

# 推理函数必须在worker函数之前定义
def extract_time(sentences):
    from torch.nn.utils.rnn import pad_sequence
    results = []
    for sentence in sentences:
        matches = re.findall(r"(\d+(\.\d+)?)", sentence)
        if matches:
            results.append(torch.tensor([float(match[0]) for match in matches]))
        else:
            results.append(torch.tensor([PAD_IDX]))
    results = pad_sequence(results, batch_first=True, padding_value=PAD_IDX)
    return results


def to_window_list_pred(pred):
    windows = np.array(list(filter(lambda x: x != PAD_IDX, pred)))
    if len(windows) == 0:
        return [[-1, -1]]
    if len(windows) % 2 != 0:
        windows = windows[:-1]
    window_list = windows.reshape(-1, 2).astype(str).tolist()
    window_list = [[float(num) for num in pair] for pair in window_list]
    if window_list == []:
        return [[-1, -1]]
    return window_list


def to_window_list_pred_vr(pred):
    windows = np.array(list(filter(lambda x: x != PAD_IDX, pred)))
    window_list = windows.astype(str).tolist()
    window_list = [float(num) for num in window_list]
    if window_list == []:
        return [-1]
    return window_list


def construct_messages_mr_fps(video_path, feature_path, fps, retrieval_segment, retrieval_mode, clip_length):
    """Constructs input message for the model based on the retrieval mode."""
    if retrieval_mode == 'mr_seg':
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"{video_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1], 
                        "feature": f"{feature_path}", "num_clips": 1, "clip_length": clip_length},
                    {"type": "text", "text": f"This is a sequence interleaved with timestamps and frames. Your task is to identify the specific timestamp(s) when the given query appears."}
                ]
            },
        ]
    elif retrieval_mode == 'mr':
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"{video_path}", "fps": fps, "video_start": retrieval_segment[0], "video_end": retrieval_segment[1]},
                    {"type": "text", "text": f"This is a sequence interleaved with timestamps and frames. Your task is to identify the temporal window (start and end timestamps) when the given query appears."}
                ]
            },
        ]
    return message


FEATURE_ROOT = os.environ.get(
    "UNITIME_FEATURE_ROOT",
    "/mnt/ali-sh-1/dataset/zeus/shiyudi/tool_rl/tmp_features",
)

def run_inference(model, processor, data, device):
    video_start = data.get("video_start", 0)
    duration = data.get("duration")
    video_end = data.get("video_end", duration)
    query = data.get("query")
    video_path = data.get("video_path")
    retrieval_segment = [video_start, video_end]

    if video_end - video_start <= 256:
        retrieval_mode = "mr"
    else:
        retrieval_mode = "mr_seg"
    
    if retrieval_mode == "mr_seg":
        feature_path = feature(model, processor, video_path, feature_root=FEATURE_ROOT)
    else:
        feature_path = None
    
    message = construct_messages_mr_fps(
        video_path, feature_path, 2, retrieval_segment, retrieval_mode, 32
    )
    messages = [message]

    # Process vision inputs with the processor
    image_inputs, video_inputs, all_timestamps_combine, feature_inputs, combine_t_list = process_vision_info(messages)

    if feature_inputs is None:
        all_timestamps_num = [[round((x + y)/2, 1) for x, y in zip(sublist[::2], sublist[1::2])] for sublist in all_timestamps_combine]
    else:
        all_timestamps_num = all_timestamps_combine
    all_timestamps = [[f"timestamp: {all_t} seconds; feature: " for all_t in sublist] for sublist in all_timestamps_num]

    message_for_query = message.copy()
    message_for_query.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Query:{query}\nAnswer: "}
            ]
        }
    )
    text = processor.apply_chat_template(
        message_for_query, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        features=feature_inputs,
        timestamps=all_timestamps,
        combine_t_list=combine_t_list,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    if feature_inputs is not None:
        feature_inputs = torch.cat(
            [feature_inputs[i].reshape(-1, feature_inputs[i].shape[3]) for i in range(len(feature_inputs))],
            dim=0
        )
    else:
        feature_inputs = None
    if combine_t_list is not None:
        combine_t_list = [torch.tensor(i) for i in combine_t_list]
    else:
        combine_t_list = None
    if 'pixel_values_videos' in inputs:
        pixel_values_videos = inputs['pixel_values_videos']
    else:
        pixel_values_videos = None 

    model_inputs = dict(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=inputs['video_grid_thw'],
            feature_inputs=feature_inputs,
            combine_t_list=combine_t_list
        )
    
    gen_kwargs = {'max_new_tokens': 128, 'temperature': 0.0, 'top_p': 1.0, 'num_beams': 1, 'do_sample': False}

    # Generate predictions
    generated_ids = model.generate(
        **model_inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        do_sample=True if gen_kwargs["temperature"] > 0 else False,
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        num_beams=gen_kwargs["num_beams"],
        max_new_tokens=gen_kwargs["max_new_tokens"],
        use_cache=True,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    predictions = extract_time(output_text).numpy()
    if retrieval_mode == 'mr':
        pred_window = to_window_list_pred(predictions[0])
    else:
        pred_relevant_windows_mr_seg = data.get("pred_relevant_windows_mr_seg", None)
        if pred_relevant_windows_mr_seg is None:
            pred_relevant_windows_mr_seg = []
        pred_relevant_windows_mr_seg.append(to_window_list_pred_vr(predictions[0]))
        data["pred_relevant_windows_mr_seg"] = pred_relevant_windows_mr_seg

        sampled_timestamps = torch.tensor(all_timestamps_num[0])
        predictions_seg = to_window_list_pred_vr(predictions[0])
        video_start = predictions_seg[0]
        video_end = predictions_seg[-1]
        duration = data["duration"]
        try:
            predict_end_index = sampled_timestamps.index(video_end)
            if predict_end_index == len(sampled_timestamps) - 1:
                video_end = duration
            else:
                video_end = sampled_timestamps[predict_end_index + 1]
        except:
            clip_length = sampled_timestamps[1] - sampled_timestamps[0]
            video_end = video_start + clip_length
        video_end = min(video_end, duration)
        video_start = max(0, video_start)

        data["video_start"] = video_start
        data["video_end"] = video_end
        return run_inference(model, processor, data, device)
    
    return pred_window


# worker进程函数 - 移到类外面
def worker_process_func(model_finetune_path, model_local_path, device, worker_id, request_queue, result_queue, is_busy, request_count):
    """独立的worker进程函数"""
    try:
        print(f"Worker {worker_id}: Starting on {device}...")
        
        # 在子进程中重新导入和设置
        import torch
        import os
        import sys
        import re
        import numpy as np
        from torch.nn.utils.rnn import pad_sequence
        
        # 子进程中的模型导入
        sys.path.append(os.environ.get('UNITIME_TOOL_DIR', '/mnt/ali-sh-1/usr/shiyudi1/REVPT/tools/UniTime'))
        from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor
        from qwen_vision_process import process_vision_info
        from feature import feature
        
        # 设置CUDA设备
        if 'cuda' in device:
            gpu_id = int(device.split(':')[1])
            torch.cuda.set_device(gpu_id)
            # 在spawn进程中，不需要手动set_device
            print(f"Worker {worker_id}: Set CUDA_VISIBLE_DEVICES={gpu_id}")
        
        print(f"Worker {worker_id}: Loading UniTime model to {device}...")
        
        model = Qwen2VLMRForConditionalGeneration.from_pretrained(
            model_finetune_path, 
            torch_dtype=torch.bfloat16, 
            device_map={"": device}, 
            attn_implementation="flash_attention_2"
        )
        model.eval()
        processor = Qwen2VLMRProcessor.from_pretrained(model_local_path)
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
                
                video_path, text_prompt, duration = request_data
                data = {
                    "video_path": video_path,
                    "duration": duration,
                    "query": text_prompt,
                    "video_start": 0,
                    "video_end": duration
                }
                
                start_time = time.time()
                pred_window = run_inference(model, processor, data, device)
                process_time = time.time() - start_time
                
                result = {
                    "pred_window": pred_window,
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


class UniTimeWorker:
    def __init__(self, model_finetune_path, model_local_path, device="cpu", worker_id=0):  
        self.device = device
        self.model_finetune_path = model_finetune_path
        self.model_local_path = model_local_path
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
                self.model_finetune_path, 
                self.model_local_path,
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

    async def process_request_async(self, video_path, text_prompt, duration):
        """异步处理请求"""
        if not self.request_queue or not self.result_queue:
            raise RuntimeError("Worker not started")
            
        request_id = str(uuid.uuid4())
        
        def _sync_process():
            try:
                self.request_queue.put((request_id, (video_path, text_prompt, duration)), timeout=10)
                
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
        self.workers: List[UniTimeWorker] = []
        self.current_index = 0
        
    def add_worker(self, worker: UniTimeWorker):
        self.workers.append(worker)
        
    def get_available_worker(self) -> Optional[UniTimeWorker]:
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
        "model_finetune_path": os.environ.get(
            "UNITIME_FINETUNE_PATH",
            "/mnt/ali-sh-1/dataset/zeus/shiyudi/models/UniTime",
        ),
        "model_local_path": os.environ.get(
            "UNITIME_BASE_PATH",
            "/mnt/ali-sh-1/dataset/zeus/shiyudi/models/Qwen2-VL-7B-Instruct",
        ),
    }
    
    # 为每个GPU创建worker
    workers_to_start = []
    for gpu_id in range(gpu_count):
        device = f"cuda:{gpu_id}" if gpu_count > 1 or torch.cuda.is_available() else "cpu"
        worker = UniTimeWorker(
            model_finetune_path=default_config["model_finetune_path"],
            model_local_path=default_config["model_local_path"],
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
    title="UniTime API",
    description="API for UniTime Temporal Grounding with Auto GPU Detection",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/temporal")
async def temporal_grounding(
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
            
            print(f"→ Request assigned to Worker {worker.worker_id} ({worker.device}) [Queue: {worker.get_queue_size()}]")
            
            result = await worker.process_request_async(video_path, text_prompt, duration)
            
            if "pred_window" in result:
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


if __name__ == "__main__":
    print("Starting UniTime API with auto GPU detection...")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        log_level="info"
    )
