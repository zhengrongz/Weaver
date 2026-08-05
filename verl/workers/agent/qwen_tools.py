import torch
import numpy as np
import requests
import io as _stdlib_io          # standard-library io (BytesIO etc.)
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
import cv2
import inspect
import sys
from PIL import ImageDraw
import decord
import ffmpeg
import asyncio
import aiohttp
import json
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import threading
import re
import math
import os
from pathlib import Path
from qwen_vl_utils import process_vision_info, smart_resize
from torchvision import io as torchvision_io, transforms
from torchvision.transforms import InterpolationMode

video_processing_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="video_processor")

def numpy_video_to_bytes_ffmpeg(raw_video: np.ndarray, fps: int = 4) -> bytes:
    """
    使用 FFmpeg 将 NumPy 视频编码为 MP4 二进制流（内存中）
    Args:
        video: np.ndarray, shape (T, H, W, C), dtype=np.uint8
        fps: 帧率
    Returns:
        bytes: MP4 二进制数据
    """
    # video = np.ascontiguousarray(raw_video.transpose(0,2,3,1))
    if len(raw_video.shape) != 4:
        video = np.expand_dims(raw_video, axis=0)
    else:
        video = raw_video
    if video.shape[-1] != 3:
        video = np.ascontiguousarray(video.transpose(0,2,3,1))
    T, H, W, C = video.shape
    if C not in [1, 3, 4]:
        raise ValueError("Channel must be 1, 3, or 4")
    
    # 确保数据类型正确
    if video.dtype != np.uint8:
        video = video.astype(np.uint8)
    
    # FFmpeg 输入参数
    if C == 1:
        pix_fmt = 'gray'
    elif C == 3:
        pix_fmt = 'rgb24'
    elif C == 4:
        pix_fmt = 'rgba'
    
    input_args = {
        'format': 'rawvideo',
        'pix_fmt': pix_fmt,
        's': f'{W}x{H}',
        'r': fps
    }
    
    # FFmpeg 输出参数
    output_args = {
        'format': 'mp4',
        'vcodec': 'libx264',
        'preset': 'fast',
        'pix_fmt': 'yuv420p',
        'movflags': 'frag_keyframe+empty_moov'  # 重要：用于流式传输
    }
    
    process = None
    try:
        # 启用 stderr 捕获
        stream = ffmpeg.input('pipe:', **input_args)
        
        # 添加缩放滤镜确保尺寸是2的倍数
        stream = stream.filter('scale', 
                            w=f"trunc({W}/2)*2",  # 确保宽度是2的倍数
                            h=f"trunc({H}/2)*2")  # 确保高度是2的倍数
        
        process = (
            stream
            .output('pipe:', **output_args)
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
        )
        
        # 写入帧数据
        video_bytes_data = video.tobytes()
        stdout, stderr = process.communicate(input=video_bytes_data)
        
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8') if stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg failed with return code {process.returncode}: {error_msg}")
        
        return stdout
        
    except Exception as e:
        if process:
            process.kill()
            process.wait()
        raise RuntimeError(f"Video encoding failed: {str(e)}")

def fetch_tool_desc() -> str:
    # Get all classes in current module
    tool_classes = []
    current_module = sys.modules[__name__]
    
    for name, obj in inspect.getmembers(current_module):
        # Check if it's a class that inherits from BaseTool and has required attributes
        if (inspect.isclass(obj) and 
            issubclass(obj, BaseTool) and 
            obj != BaseTool):
            tool_classes.append(obj)
    
    tool_prompts = []
    for tool_class in tool_classes:
        tool_prompts.append(str(tool_class().function))
    
    return '\n'.join(tool_prompts)

def fetch_tools(placeholder='<|vision_start|><|image_pad|><|vision_end|>', tool_urls: Dict[str, str] = None) -> Dict:
    # Get all classes in current module
    tools = {}
    current_module = sys.modules[__name__]
    tool_urls = tool_urls or {}
    
    for name, obj in inspect.getmembers(current_module):
        # Check if it's a class that inherits from BaseTool and has required attributes
        if (inspect.isclass(obj) and
            issubclass(obj, BaseTool) and
            obj != BaseTool):
            # Inspect __init__ signature to pass url/urls if supported
            init_sig = inspect.signature(obj.__init__)
            init_kwargs = {'placeholder': placeholder}
            if 'url' in init_sig.parameters:
                tool_key = name  # e.g. 'TemporalGroundingTool'
                if tool_key in tool_urls:
                    init_kwargs['url'] = tool_urls[tool_key]
            if 'frame_url' in init_sig.parameters:
                if 'FrameSelectionTool_frame' in tool_urls:
                    init_kwargs['frame_url'] = tool_urls['FrameSelectionTool_frame']
            if 'temporal_url' in init_sig.parameters:
                if 'FrameSelectionTool_temporal' in tool_urls:
                    init_kwargs['temporal_url'] = tool_urls['FrameSelectionTool_temporal']
            tool_instance = obj(**init_kwargs)
            tools[tool_instance.name] = tool_instance
    
    return tools


def highlight_video(video, bboxes):
    new_video = np.copy(video)
    if new_video.dtype != np.uint8:
        new_video = new_video.astype(np.uint8)
    
    if len(new_video.shape) != 4:
        new_video = np.expand_dims(new_video, axis=0)
    if new_video.shape[-1] != 3:
        new_video = np.ascontiguousarray(new_video.transpose(0, 2, 3, 1))
    h, w = new_video.shape[1:3]
    flags = []
    # import ipdb; ipdb.set_trace()
    for i in range(len(bboxes)):
        bbox_dict = bboxes[i]
        # import ipdb; ipdb.set_trace()
        keys = list(bbox_dict.keys())
        flag = False
        for key in keys:
            bbox_list = bbox_dict[key]
            offset = new_video.shape[0] - len(bbox_list)
            # for j in range(len(bbox_list)):
            for j in range(len(bbox_list)):
                frame = new_video[j+offset]
                # if not isinstance(bbox_list, list):
                bbox = bbox_list[j]
                
                # import ipdb; ipdb.set_trace()
            # bbox = bboxes[i]
                # if bbox == [-1,-1,-1,-1]:
                if bbox == [0,0,0,0]:
                    continue
                x1, y1, x2, y2 = bbox
            
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                flag = True
            
            # 更新视频帧
                new_video[j+offset] = frame
        flags.append(flag)
    new_video = np.ascontiguousarray(new_video.transpose(0,3,1,2))
    
    return new_video, flags


def time_to_seconds(time_str):
    if not time_str:
        return 0.0

    parts = time_str.split(':')
    
    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            total_seconds = minutes * 60 + seconds
            
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            total_seconds = hours * 3600 + minutes * 60 + seconds
            
        else:
            return 0.0
            
        return total_seconds
        
    except (ValueError, TypeError):
        return 0.0

def match_time(query):
    pattern = r'(from|between)\s+(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)\s+(to|and)\s+(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)'
    
    matches = re.finditer(pattern, query, re.IGNORECASE)
    results = []
    
    for match in matches:
        prefix = match.group(1).lower()
        time1_str = match.group(2)
        connector = match.group(3).lower()
        time2_str = match.group(4)
        
        parts1 = time1_str.split(':')
        parts2 = time2_str.split(':')
        
        if len(parts1) not in [2, 3] or len(parts2) not in [2, 3]:
            continue

        if prefix == 'from' and connector == 'to':
            pattern_type = "from-to"
        elif prefix == 'between' and connector == 'and':
            pattern_type = "between-and"
        else:
            continue
        
        time1_seconds = time_to_seconds(time1_str)
        time2_seconds = time_to_seconds(time2_str)
            
        results.append([time1_seconds, time2_seconds])

    
    if results != []:
        flag = True
    else:
        flag = False
    
    return results, flag

async def highlight_video_async(video, bboxes):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        video_processing_pool,
        lambda: highlight_video(video, bboxes)
    )

def read_video(video_path, start=None, end=None, target_fps=1, max_frames=None):
    path = Path(video_path)
    origin_path = Path(*path.parts[:7])
    filename = os.path.basename(video_path)
    file_id = os.path.splitext(filename)[0]
    frame_dir = os.path.join(origin_path, "frames", file_id)

    if os.path.exists(frame_dir) and max_frames is None:
        frame_1fps_dir = os.path.join(frame_dir, "1fps.pt")
        video_frames = torch.load(frame_1fps_dir)
        raw_video = video_frames['video'][0].numpy()
        if start is not None:
            real_start = max(0, int(start))
        else:
            real_start = 0
        if end is not None:
            real_end = min(raw_video.shape[0] - 1, math.ceil(end))
        else:
            real_end = raw_video.shape[0] - 1
        if real_start >= real_end:
            if real_end == raw_video.shape[0] - 1:
                real_start = real_end - 1
            else:
                real_end = real_start + 1
        # import ipdb; ipdb.set_trace()
        frames = raw_video[real_start:real_end]
    else:
        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        
        if start is not None:
            start_frame = int(start * original_fps)
        else:
            start_frame = 0
        if end is not None:
            end_frame = int(end * original_fps)
        else:
            end_frame = total_frames - 1
        
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames - 1))
        
        if start_frame >= end_frame:
            start_frame = max(0, end_frame - 1)
            end_frame = start_frame + 1
        
        frame_interval = int(original_fps / target_fps)
        frame_indices = np.arange(start_frame, end_frame + 1, frame_interval)
        
        frame_indices = frame_indices[frame_indices < total_frames]
        

        if len(frame_indices) == 0:
            frame_indices = np.array([start_frame])

        if max_frames:
            if len(frame_indices) > max_frames:
                frame_indices = np.linspace(start_frame, end_frame, max_frames, dtype=np.int32)
        
        frames = vr.get_batch(frame_indices).asnumpy()

    video_tensor = torch.tensor(frames)
    if video_tensor.shape[-1] == 3:
        video_tensor = video_tensor.permute(0,3,1,2)
    height, width = video_tensor.shape[-2:]
    resized_height, resized_width = smart_resize(
            height,
            width,
            factor=14,
            min_pixels=128*28*28,
            max_pixels=448*448,
        )
    video = transforms.functional.resize(
        video_tensor,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float().permute(0,2,3,1).numpy()

    if (video < 0).sum() > 0:
        video = transforms.functional.resize(
        video_tensor,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    ).float().permute(0,2,3,1).numpy()

    video = np.clip(video, 0., 255.).astype(int)
    # print(video.shape)
    # if video.shape[-1] != 3 or video.shape[0] == 1 or len(video.shape) == 3:
    # import ipdb; ipdb.set_trace()

    
    return video
    

class TemporalGroundingTool(BaseTool):
    name = "temporal_grounding"
    description = "Temporal grounding tool. It returns relevant temporal winodw of the event you take care of." \
                  "The grounding is not perfect , it may wrongly ground timestamps not relative to the event. You should use the output as a reference , not as a ground truth."
            
    parameters = [
        {
            'name': 'video_path',
            'type': 'str',
            'description': "The video path to load for grounding",
            'required': True,
        },
        {
            'name': 'query',
            'type': 'str',
            'description': "The event to ground in the video.",
            'required': True,
        }
    ]

    def __init__(self, url='http://localhost:9995/temporal', placeholder='<|vision_start|><|video_pad|><|vision_end|>'):
        self.url = url
        self.placeholder = placeholder
    

    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            video_path = args["video_path"]
            query = args.get("query", None)
            duration = args['duration']

            data = {
                "video_path": video_path,
                "text_prompt": query,
                "duration": duration
            }
            # try:
            #     temporal_window, flag = match_time(query)
            #     if flag:
            #         temporal_window = temporal_window[0]
            # except:
            #     flag = False
            
            # if not flag:
            # import ipdb; ipdb.set_trace()
            response = requests.post(self.url, data=data)
            response.raise_for_status()
            if response.headers.get('content-type') == 'application/json':
                json_response = response.json()
                
                if "message" in json_response:
                    return {
                        "text": "Wrong!",
                        "image": None
                    }
                    
                temporal_window = json_response.get('pred_window', [])
                if temporal_window:
                    temporal_window = temporal_window[0]
            else:
                return {
                    "text": "Unexpected response type. Cannot process result.",
                    "video": None
                }
            text = f"Ground target video clip from {temporal_window[0]}s to {temporal_window[1]}s."
            video = read_video(video_path, start=temporal_window[0], end=temporal_window[1])

            return {
                "text": text,
                "video": video,
            }
        # except:
        #     import ipdb; ipdb.set_trace()
        except KeyError as e:
            return {
                "text": f"Failed to ground video {args['video_path']} due to error: Key error: {str(e)}. Please check the tool_call format.",
                "video": None
            }
        except Exception as e:
            return {
                "text": f"Failed to ground for video {args['video_path']} due to error: {str(e)}",
                "video": None
            }



class TrackingTool(BaseTool):
    name = "spatial_tracking"
    description = "Video Tracking tool. It returns relevant bounding boxes of objects in videos" \
                  "The tracking result is not perfect , it may wrongly track the objects. You should use the output as a reference , not as a ground truth."
            
    parameters = [
        {
            'name': 'video_id',
            'type': 'str',
            'description': "The video id to load for grounding",
            'required': True,
        },
        {
            'name': 'objects',
            'type': 'List[str]',
            'description': "The objects to track in the video.",
            'required': True,
        }
    ]

    def __init__(self, url='http://localhost:9999/tracking', placeholder='<|vision_start|><|video_pad|><|vision_end|>'):
        self.url = url
        self.placeholder = placeholder
        self.gpu_ids = [4,5,6,7]
        self.gpu_counter = 0
        self.lock = threading.Lock()  # 正确初始化线程锁
        self.aio_session = None
    
    def get_next_gpu_id(self):
        """轮询获取下一个GPU ID"""
        with self.lock:
            gpu_id = self.gpu_ids[self.gpu_counter % len(self.gpu_ids)]
            self.gpu_counter += 1
            return gpu_id

    async def init_aio_session(self):
        """初始化aiohttp session"""
        if self.aio_session is None or self.aio_session.closed:
            self.aio_session = aiohttp.ClientSession()
    
    async def close_aio_session(self):
        """关闭aiohttp session"""
        if self.aio_session and not self.aio_session.closed:
            await self.aio_session.close()
    
    async def async_call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        """异步调用版本（支持多GPU）"""
        # try:
        video = env_state['video']
        objs = args['objects']
        
        if isinstance(video, list):
            video = video[0]
        
        video_bytes = numpy_video_to_bytes_ffmpeg(video)
        gpu_id = self.get_next_gpu_id()
        
        headers = {"X-GPU-ID": str(gpu_id)}
        # data = {"text_prompt": objs}
        form_data = aiohttp.FormData()
        form_data.add_field('text_prompt', json.dumps(objs))
        form_data.add_field('file', 
                           video_bytes, 
                           filename='video.mp4',
                           content_type='video/mp4')
        # files = {'file': ('video.mp4', video_bytes, 'video/mp4')}
        
        await self.init_aio_session()
        
        async with self.aio_session.post(
            self.url, data=form_data, headers=headers
        ) as response:
            response.raise_for_status()
            
            if response.content_type == 'application/json':
                json_response = await response.json()
                
                if "message" in json_response:
                    return {
                        "text": f"No objects matching '{objs}' detected.",
                        "video": None
                    }
                
                bboxes = json_response.get('pred_bboxes', [])
                text = "New visual information."
                highlighted_video = await highlight_video_async(video, bboxes)
                
                return {
                    "text": text,
                    "video": highlighted_video
                }
            else:
                return {
                    "text": "Unexpected response type",
                    "video": None
                }
                    
    
    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        # try:
        video = env_state['video']#[args['video_id']]
        # import ipdb; ipdb.set_trace()
        objs = args['objects']
        # video_p

        if isinstance(video, list):
            video = video[-1] #T,C,H,W
        
        video_bytes = numpy_video_to_bytes_ffmpeg(video)
        
        data = {
            "text_prompt": objs,
        }
        files  = {'file': ('video.mp4', video_bytes, 'video/mp4')}
        response = requests.post(self.url, files=files, data=data)
        response.raise_for_status()
        if response.headers.get('content-type') == 'application/json':
            json_response = response.json()
            
            if "message" in json_response:
                return {
                    "text": "Wrong!",
                    "video": None
                }
                
            bboxes = json_response.get('pred_bboxes', [])
            # import ipdb; ipdb.set_trace()
            highlighted_video, flags = highlight_video(video, bboxes)
            detected_objs = []
            for i, flag in enumerate(flags):
                if flag:
                    detected_objs.append(objs[i])
            if len(detected_objs) != 0:
                text = "Detect objects"
                for obj in detected_objs:
                    text += f" {obj}"
            else:
                text = "Do not detect objects, please rethink the tool usage."

            return {
                "text": text,
                "video": highlighted_video
            }
        else:
            return {
                "text": "Unexpected response type. Cannot process result.",
                "video": None
            }




class FrameSelectionTool(BaseTool):
    name = "frame_selection"
    description = "Frame selection tool. It returns relevant frame of the event you take care of." \
                  "The grounding is not perfect , it may wrongly ground timestamps not relative to the event. You should use the output as a reference , not as a ground truth."
            

    def __init__(self, frame_url='http://localhost:9997/frame', temporal_url='http://localhost:9995/temporal',placeholder='<|vision_start|><|video_pad|><|vision_end|>'):
        self.frame_url = frame_url
        self.temporal_url = temporal_url
        self.placeholder = placeholder
    

    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        video_path = args["video_path"]
        query = args.get("query", None)
        duration = args['duration']

        data = {
            "video_path": video_path,
            "text_prompt": query,
            "duration": duration
        }
        response = requests.post(self.temporal_url, data=data)
        response.raise_for_status()
        if response.headers.get('content-type') == 'application/json':
            json_response = response.json()
                
            temporal_window = json_response.get('pred_window', [])
            if temporal_window:
                temporal_window = temporal_window[0]
            
            video = read_video(video_path, start=temporal_window[0], end=temporal_window[1])

    
        video_bytes = numpy_video_to_bytes_ffmpeg(video)
        
        data = {
            "text_prompt": query,
        }
        files  = {'file': ('video.mp4', video_bytes, 'video/mp4')}
        response = requests.post(self.frame_url, files=files, data=data)
        response.raise_for_status()
        if response.headers.get('content-type') == 'application/json':
            json_response = response.json()
            
                
            frame_id = json_response.get("frame_idx", 0)
            # import ipdb; ipdb.set_trace()
            frame = np.expand_dims(video[frame_id], axis=0)
            frame_time = round((temporal_window[1]-temporal_window[0])*(frame_id/video.shape[0])+temporal_window[0],2)

            text = f"Select frame at {frame_time}s."
            # import ipdb; ipdb.set_trace()
            return {
                "text": text,
                "video": frame
            }
        else:
            return {
                "text": "Unexpected response type. Cannot process result.",
                "video": None
            }


class SpatialGroundingTool(BaseTool):
    """
    Spatial grounding tool backed by a Grounding DINO service.
    The service runs per-frame object detection and returns bounding boxes
    for each frame, which are then drawn onto the video via highlight_video().

    Default endpoint: POST http://localhost:9998/grounding
    Override via agent config: spatial_grounding_url
    """
    name = "spatial_grounding"
    description = (
        "Spatial grounding tool. Detects and localises objects in each frame of the video "
        "using Grounding DINO. Returns the video with bounding boxes drawn around detected objects. "
        "The detection is not perfect; treat the output as a reference, not ground truth."
    )
    parameters = [
        {
            'name': 'objects',
            'type': 'List[str]',
            'description': "List of object names to detect in the video.",
            'required': True,
        }
    ]

    def __init__(self, url='http://localhost:9998/grounding', placeholder='<|vision_start|><|video_pad|><|vision_end|>'):
        self.url = url
        self.placeholder = placeholder

    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        video = env_state['video']
        objs = args['objects']

        if isinstance(video, list):
            video = video[-1]  # take the most recent clip

        # Encode video/image to bytes for upload
        if len(video.shape) == 4:
            # video: (T, H, W, 3) or (T, C, H, W)
            video_bytes = numpy_video_to_bytes_ffmpeg(video)
            files = {'file': ('video.mp4', video_bytes, 'video/mp4')}
        else:
            # single image: (H, W, 3)
            img_byte_arr = _stdlib_io.BytesIO()
            image = Image.fromarray(video.astype(np.uint8))
            image.save(img_byte_arr, format='JPEG')
            files = {'file': ('image.jpeg', img_byte_arr.getvalue(), 'image/jpeg')}

        # text_prompt is a JSON-encoded list of object names
        data = {'text_prompt': json.dumps(objs)}

        try:
            response = requests.post(self.url, files=files, data=data, timeout=120)
            response.raise_for_status()
        except Exception as e:
            return {
                "text": f"Grounding DINO service error: {str(e)}",
                "video": None,
            }

        if response.headers.get('content-type', '').startswith('application/json'):
            json_response = response.json()

            if "message" in json_response:
                return {
                    "text": f"No objects matching {objs} detected.",
                    "video": None,
                }

            bboxes = json_response.get('pred_bboxes', [])
            obj_nums = json_response.get('obj_nums', {})

            highlighted_video, flags = highlight_video(video, bboxes)

            detected = [obj for obj, cnt in obj_nums.items() if cnt > 0]
            if detected:
                text = "Detect " + ", ".join(detected) + "."
            else:
                text = "Do not detect objects, please rethink the tool usage."

            return {
                "text": text,
                "video": highlighted_video,
            }
        else:
            return {
                "text": "Unexpected response type from grounding service.",
                "video": None,
            }


class TrimTool(BaseTool):
    name = "trim"
    description = "trim tool. It returns relevant temporal winodw of the event you take care of." \
                  "The grounding is not perfect , it may wrongly ground timestamps not relative to the event. You should use the output as a reference , not as a ground truth."
            

    def __init__(self, placeholder='<|vision_start|><|video_pad|><|vision_end|>'):
        self.placeholder = placeholder
    

    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            video_path = args["video_path"]
            duration = args['duration']
            start = args.get("start", 0)
            end = args.get("end", int(duration))
            # import ipdb; ipdb.set_trace()
            temporal_window = [start, end]

            text = f"Ground target video clip from {temporal_window[0]}s to {temporal_window[1]}s."
            video = read_video(video_path, start=temporal_window[0], end=temporal_window[1], max_frames=64)
            # print(temporal_window)

            return {
                "text": text,
                "video": video,
            }
        # except:
        #     import ipdb; ipdb.set_trace()
        except KeyError as e:
            return {
                "text": f"Failed to ground video {args['video_path']} due to error: Key error: {str(e)}. Please check the tool_call format.",
                "video": None
            }
        except Exception as e:
            return {
                "text": f"Failed to ground for video {args['video_path']} due to error: {str(e)}",
                "video": None
            }


class TemporalCountTool(BaseTool):
    name = "temporal_count"
            

    def __init__(self, url='http://localhost:9997/count', placeholder='<|vision_start|><|video_pad|><|vision_end|>'):
        self.url = url
        self.placeholder = placeholder
    

    def call(self, args: Dict[str, Any], env_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            video_path = args["video_path"]

            duration = args['duration']

            query = args.get("query", None)

            data = {
                "video_path": video_path,
                "text_prompt": query,
                "duration": duration
            }
            response = requests.post(self.url, data=data)
            response.raise_for_status()
            if response.headers.get('content-type') == 'application/json':
                json_response = response.json()
                
                    
                temporal_windows = json_response.get('pred_windows', [[[0, duration]]])
                text = f"Count {len(temporal_windows)} clips for the query and concat all clips successfully."
                if temporal_windows == []:
                    temporal_windows = [[0, int(duration)]]
                    text = f"Do not count {query} in video, return sampled raw video."
            else:
                return {
                    "text": "Unexpected response type. Cannot process result.",
                    "video": None
                }
            # import ipdb; ipdb.set_trace()
            video = []
            # print(temporal_windows)
            for temporal_window in temporal_windows:
                # if temporal_window:
                #     temporal_window = temporal_window[0]
                sub_video = read_video(video_path, start=temporal_window[0], end=temporal_window[1], max_frames=64)
                video.append(sub_video)
            
            if len(video) != 1:
                video = np.concatenate(video, axis=0)
            else:
                video = video[0]
            # print(temporal_window)

            return {
                "text": text,
                "video": video,
            }
        except KeyError as e:
            return {
                "text": f"Failed to ground video {args['video_path']} due to error: Key error: {str(e)}. Please check the tool_call format.",
                "video": None
            }
        except Exception as e:
            return {
                "text": f"Failed to ground for video {args['video_path']} due to error: {str(e)}",
                "video": None
            }