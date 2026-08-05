import torch
import sys
import json
from tqdm import *
import decord
import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_path = os.path.join(current_dir, 'UniTime')
sys.path.append(subfolder_path)
subfolder_path = os.path.join(current_dir, 'GroundedSAM2')
sys.path.append(subfolder_path)
import sys
import time
import io
import tempfile
import multiprocessing as mp
import matplotlib.pyplot as plt
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from tqdm import tqdm
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import re
import ast
import cv2
import copy

from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor
from qwen_vision_process import process_vision_info
from feature import feature
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

PAD_IDX = -100

def save_video(video, save_dir):
    num_frames, height, width, channels = video.shape
    
    # 设置视频编码器和参数（使用MP4格式）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # 帧率，可根据需要调整
    
    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(save_dir, fourcc, fps, (width, height))
    
    # 逐帧写入视频
    for i in range(num_frames):
        frame = video[i]
        # 将 RGB 转换为 BGR (OpenCV 使用 BGR 格式)
        frame_bgr = cv2.cvtColor(np.uint8(frame), cv2.COLOR_RGB2BGR)
        
        # 写入帧
        out.write(frame_bgr)
    
    # 释放 VideoWriter
    out.release()

def time_to_seconds(time_str):
    """
    将时间字符串转换为秒数（float类型）
    支持格式：MM:SS 和 HH:MM:SS
    
    参数:
    time_str (str): 时间字符串，如 "05:30" 或 "01:23:45.5"
    
    返回:
    float: 对应的秒数
    """
    if not time_str:
        return 0.0
    
    # 分割时间部分
    parts = time_str.split(':')
    
    try:
        if len(parts) == 2:
            # MM:SS 格式
            minutes = int(parts[0])
            seconds = float(parts[1])
            total_seconds = minutes * 60 + seconds
            
        elif len(parts) == 3:
            # HH:MM:SS 格式
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            total_seconds = hours * 3600 + minutes * 60 + seconds
            
        else:
            # 无效格式
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
        
        # 验证时间格式（必须是2或3个部分）
        parts1 = time1_str.split(':')
        parts2 = time2_str.split(':')
        
        if len(parts1) not in [2, 3] or len(parts2) not in [2, 3]:
            continue
        
        # 确定句式类型
        if prefix == 'from' and connector == 'to':
            pattern_type = "from-to"
        elif prefix == 'between' and connector == 'and':
            pattern_type = "between-and"
        else:
            continue
        
        # 转换为秒数
        time1_seconds = time_to_seconds(time1_str)
        time2_seconds = time_to_seconds(time2_str)
            
        results.append([time1_seconds, time2_seconds])

    
    if results != []:
        flag = True
    else:
        flag = False
    
    return results, flag

def highlight_video(video, bboxes):
    new_video = np.copy(video)
    h, w = video.shape[1:3]
    for i in range(len(bboxes)):
        bbox_dict = bboxes[i]
        # import ipdb; ipdb.set_trace()
        keys = list(bbox_dict.keys())
        for key in keys:
            bbox_list = bbox_dict[key]
            offset = video.shape[0] - len(bbox_list)
            # for j in range(len(bbox_list)):
            for j in range(bbox_list.shape[0]):
                frame = new_video[j+offset]
                bbox = bbox_list[j].tolist()
                # import ipdb; ipdb.set_trace()
            # bbox = bboxes[i]
                # if bbox == [-1,-1,-1,-1]:
                if bbox == [0,0,0,0]:
                    continue
                x1, y1, x2, y2 = bbox
            # frame = new_video[i]
            # frame_np = frame.transpose(1, 2, 0)  # (H, W, C)
            # frame_np = (frame_np).astype(np.uint8)  # 假设 tensor 值在 [0,1] 范围内
            
            # 绘制矩形框
            # import ipdb; ipdb.set_trace()
                # cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 更新视频帧
                new_video[j+offset] = frame
    
    return new_video

def find_boundaries_torch(mask):
    from skimage.segmentation import find_boundaries
    image_data = np.where(mask, 255, 0).astype(np.uint8)
    # cv2.imwrite('binary_image.png', image_data)
    # import ipdb; ipdb.set_trace()
    mask_np = mask.to(torch.bool).numpy()
    boundaries = find_boundaries(mask_np, mode='outer')
    boundary_points = np.argwhere(boundaries)
    if boundary_points.size == 0:
        return torch.tensor([-1, -1, -1, -1], dtype = torch.bfloat16)
    h0, w0 = boundary_points.min(axis=0)
    h1, w1 = boundary_points.max(axis=0)
    return torch.tensor([w0 / mask.shape[1], h0 / mask.shape[0],  w1 / mask.shape[1], h1 / mask.shape[0]], dtype = torch.bfloat16)

def get_video(video_path, start=None, end=None, num_frames=64):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    fps = vr.get_avg_fps()

    if start is not None:
        start_frame = max(0, min(total_frames-1, int(start * fps)))
    else:
        start_frame = 0
    if end is not None:
        end_frame = max(0, min(total_frames-1, int(end * fps)))
    else:
        end_frame = total_frames - 1
    
    # 确保帧索引在有效范围内
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(0, min(end_frame, total_frames - 1))

    if start_frame >= end_frame:
        start_frame = end_frame - 1
        if start_frame < 0:
            start_frame = 0
            end_frame = 1
        # raise ValueError(f"start frame {start_frame} >= end frame {end_frame}!")
    
    # 计算要采样的帧索引（均匀间隔）
    frame_indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int)
    
    # 读取指定帧
    frames = vr.get_batch(frame_indices).asnumpy()

    return frames

def extract_time(sentences):
    results = []
    for sentence in sentences:
        matches = re.findall(r"(\d+(\.\d+)?)", sentence)
        if matches:
            results.append(torch.tensor([float(match[0]) for match in matches]))
        else:
            results.append(torch.tensor([PAD_IDX]))
    results = pad_sequence(results,batch_first=True,padding_value=PAD_IDX)
    return results

def to_window_list_pred(pred):
    windows = np.array(list(filter(lambda x: x != PAD_IDX, pred)))
    if len(windows) == 0:
        return [[-1,-1]]
    if len(windows) % 2 != 0:
        windows = windows[:-1]
    window_list = windows.reshape(-1, 2).astype(str).tolist()
    window_list = [[float(num) for num in pair] for pair in window_list]
    if window_list == []:
        return [[-1,-1]]
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

class UniTime:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_finetune_path = "/mnt/ali-sh-1/dataset/zeus/shiyudi/models/UniTime"
        model_local_path = "/mnt/ali-sh-1/dataset/zeus/shiyudi/models/Qwen2-VL-7B-Instruct"
        self.model = Qwen2VLMRForConditionalGeneration.from_pretrained(model_finetune_path, torch_dtype=torch.bfloat16, device_map={"": self.device}, attn_implementation="flash_attention_2")
        self.model.eval()
        self.processor = Qwen2VLMRProcessor.from_pretrained(model_local_path)

    def forward(self, video_path, query, duration):

        data = {
            "video_path": video_path,
            "duration": duration,
            "query": query,
            "video_start": 0,
            "video_end": duration
        }

        pred_window = self.run_inference(data)

        return pred_window
    def run_inference(self, data):
        device = self.model.visual.device
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
            feature_path = feature(self.model, self.processor, video_path, feature_root="/mnt/ali-sh-1/dataset/zeus/shiyudi/tool_rl/sub_video_feature")
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
        text = self.processor.apply_chat_template(
            message_for_query, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
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
        generated_ids = self.model.generate(
            **model_inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
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
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


        predictions = extract_time(output_text).numpy()
        if retrieval_mode =='mr':
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
            return self.run_inference(data)
        
        return pred_window


class GroundedSAM2:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam2_checkpoint = "/mnt/ali-sh-1/dataset/zeus/shiyudi/models/sam2/sam2.1_hiera_large.pt"
        self.model_cfg = "//mnt/ali-sh-1/usr/shiyudi1/REVPT/tools/GroundedSAM2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

        self.model_id = "/mnt/ali-sh-1/dataset/zeus/shiyudi/models/grounding-dino-base"

        # if self.device.startswith("cuda:"):
        #     gpu_id = int(self.device.split(":")[-1])
        #     torch.cuda.set_device(gpu_id)

        # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        # if torch.cuda.get_device_properties(0).major >= 8:
        #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     torch.backends.cudnn.allow_tf32 = True

        self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint)
        self.sam2_image_model = build_sam2(self.model_cfg, self.sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(self.sam2_image_model)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean, std)
        type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
        self.transform = transforms.Compose(
            [   
                type_transform,
                transforms.Resize(
                    (1024, 1024),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                normalize,
            ]
        )
    
    def forward(self, video_numpy, objects):
        height, width = video_numpy.shape[1:3]

        video = torch.tensor(video_numpy, device=self.device)

        transformed_video = self.transform(video.permute(0,3,1,2))
        transformed_video = transformed_video.to(torch.bfloat16) 
        transformed_video = transformed_video.unsqueeze(0)

        inference_state = self.video_predictor.init_state_images(images=transformed_video, video_height=height, video_width=width)
        total_bboxes = []

        sam2_masks = MaskDictionaryModel()
        PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
        objects_count = 0
        for obj in objects:
            if not obj.endswith('.'):
                obj = obj + "."
            obj = obj.lower()
            # obj = "bag."
            total_video_segments = []
            for frame_idx in range(0, video_numpy.shape[0], 4):
                ann_image = Image.fromarray(video_numpy[frame_idx])
                mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{obj}.npy")

                inputs = self.processor(images=ann_image, text=obj, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.grounding_model(**inputs)

                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[ann_image.size[::-1]]
                )

                self.image_predictor.set_image(np.array(ann_image.convert("RGB")))

                # process the detection results
                input_boxes = results[0]["boxes"] # .cpu().numpy()
                # print("results[0]",results[0])
                OBJECTS = results[0]["labels"]
                if input_boxes.shape[0] != 0:
                    # prompt SAM 2 image predictor to get the mask for the object
                    masks, scores, logits = self.image_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_boxes,
                        multimask_output=False,
                    )
                    # convert the mask shape to (n, H, W)
                    if masks.ndim == 2:
                        masks = masks[None]
                        scores = scores[None]
                        logits = logits[None]
                    elif masks.ndim == 4:
                        masks = masks.squeeze(1)
                    
                    if mask_dict.promote_type == "mask":
                        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(self.device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
                    else:
                        raise NotImplementedError("SAM 2 video predictor only support mask prompts")


                    """
                    Step 4: Propagate the video predictor to get the segmentation results for each frame
                    """
                    objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
                    print("objects_count", objects_count)
                else:
                    mask_dict = sam2_masks

                if len(mask_dict.labels) == 0:
                    # mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list = frame_names[frame_idx:frame_idx+step])
                    # print("No object detected in the frame, skip the frame {}".format(frame_idx))
                    continue
                else: 
                    self.video_predictor.reset_state(inference_state)

                    for object_id, object_info in mask_dict.labels.items():
                        frame_idx, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                                inference_state,
                                frame_idx,
                                object_id,
                                object_info.mask,
                            )
        
                    video_segments = {}  # output the following {step} frames tracking masks
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=3, start_frame_idx=frame_idx):
                        frame_masks = MaskDictionaryModel()
                        
                        for i, out_obj_id in enumerate(out_obj_ids):
                            out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                            object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
                            object_info.update_box()
                            frame_masks.labels[out_obj_id] = object_info
                            # image_base_name = frame_names[out_frame_idx].split(".")[0]
                            frame_masks.mask_name = f"mask_{obj}.npy"
                            frame_masks.mask_height = out_mask.shape[-2]
                            frame_masks.mask_width = out_mask.shape[-1]

                        video_segments[out_frame_idx] = frame_masks
                        sam2_masks = copy.deepcopy(frame_masks)
                    
                    total_video_segments.append(video_segments)
                
            bboxes = {}
            # import ipdb; ipdb.set_trace()
            for video_segment in total_video_segments:
                key_list = list(video_segment.keys())
                for key in key_list:
                    labels = video_segment[key].labels
                    label_key_list = list(labels.keys())
                    for label_key in label_key_list:
                        if label_key not in bboxes:
                            bboxes[label_key] = np.zeros((video_numpy.shape[0], 4))
                        
                        label = labels[label_key]
                        # import ipdb; ipdb.set_trace()
                        bbox = np.array([label.x1, label.y1, label.x2, label.y2])
                        bboxes[label_key][key] = bbox


            #     bbox = [round(video_segment.labels[1].x1/width, 3), round(video_segment.labels[1].y1/height, 3), round(video_segment.labels[1].x2/width, 3), round(video_segment.labels[1].y2/height, 3)]
            #     bboxes.append(bbox)
            # import ipdb; ipdb.set_trace()
            total_bboxes.append(bboxes)

        return total_bboxes

        

temporal_tool = UniTime()
spatial_tool = GroundedSAM2()
# import ipdb; ipdb.set_trace()
datas = json.load(open("/mnt/ali-sh-1/usr/shiyudi1/multimodal_cot/datas/stage2/data/longvideo_qac_tool_part.json"))

new_datas = []
for data in tqdm(datas[4200:]):
    new_data = {}
    video_path = data['video_path']
    video = get_video(video_path)
    qac = data['qac']
    pattern = r'<tool_call>(.*?)</tool_call>'
    thinking_process = qac['Thinking Process']
    question = qac['Question']
    options = qac['Options']
    if "A" not in options or "B" not in options or "C" not in options or "D" not in options:
        continue
    answer = qac['Correct Answer']
    video_id = data['id']
    duration = data['duration']
    prompt = f"<video>\nQuestion: {question}\nOptions:\n(A) {options['A']}\n(B) {options['B']}\n(C) {options['C']}\n(D) {options['D']}\nThink step by step and call tools if needed, then answer."
    prompt_content = {
        "from": "human",
        "value": prompt
    }
    conversations = [prompt_content]
    videos = [video_path]
    video_num = 0
    video_base_dir = os.path.join("/mnt/ali-sh-1/dataset/zeus/shiyudi/tool_rl/think_with_videos/longvideo", video_id)
    os.makedirs(video_base_dir, exist_ok=True)
    # flag = False
    # for step in thinking_process:
    #     content = step['content']
    #     if "spatial_tracking" in content:
    #         flag = True
    #         break
    # if flag == True:
    #     continue

    for step in thinking_process:
        content = step['content']

        matches = re.finditer(pattern, content, re.DOTALL)
        matches_list = list(matches)

        if matches_list:
            for i, match in enumerate(matches_list):
                tool_call_content = match.group(1).strip()
                try:
                    tool_call_dict = ast.literal_eval(tool_call_content)
                except:
                    continue
                called_tool = tool_call_dict['tool_name']
                if called_tool == "temporal_grounding":
                    query = tool_call_dict['query']
                    temporal_window, flag = match_time(query)
                    if not flag:
                        temporal_window = temporal_tool.forward(video_path, query, duration) #todo
                    temporal_window = temporal_window[0]
                    # import ipdb; ipdb.set_trace()
                    grounded_video = get_video(video_path, start=temporal_window[0], end=temporal_window[1], num_frames=32)
                    save_video_dir = os.path.join(video_base_dir, f"{video_num}.mp4")
                    save_video(grounded_video, save_video_dir)
                    video_num += 1
                    videos.append(save_video_dir)
                    new_text = "<video>\nNew visual information. Continue analyzing."

                if called_tool == "spatial_tracking":
                    # import ipdb; ipdb.set_trace()
                    objs = tool_call_dict['objects']
                    bboxes = spatial_tool.forward(video, objs)
                    highlighted_video = highlight_video(video, bboxes)
                    save_video_dir = os.path.join(video_base_dir, f"{video_num}.mp4")
                    save_video(highlighted_video, save_video_dir)
                    video_num += 1
                    videos.append(save_video_dir)
                    new_text = "<video>\nNew visual information. Continue analyzing."

            new_pattern = r'(?=<tool_call>)|(?<=</tool_call>)'
            parts = re.split(new_pattern, content)

            # 过滤空字符串并去除空白
            results = [part.strip() for part in parts if part.strip()]
            for result in results:
                if "<tool_call>" in result:
                    new_step = {
                        "from": "gpt",
                        "value": result
                    }
                    conversations.append(new_step)
                    new_human_step = {
                        "from": "human",
                        "value": new_text
                    }
                    conversations.append(new_human_step)
                else:
                    new_step = {
                        "from": "gpt",
                        "value": result
                    }
                    conversations.append(new_step)
        else:
            new_step = {
                "from": "gpt",
                "value": content
            }
            conversations.append(new_step)

    new_data['video'] = videos
    new_data['conversations'] = conversations
    new_datas.append(new_data)
    # import ipdb; ipdb.set_trace()

    

with open("/mnt/ali-sh-1/dataset/zeus/shiyudi/tool_rl/think_with_videos/longvideo_sft_train_tool_part_7.json", "w") as f:
    json.dump(new_datas, f, indent=2)







# \n\n<tool_call>{\n  \"tool_name\": \"temporal_grounding\",\n  \"query\": \"man wearing a jacket with 'Auckland Zoo' on it between 0:00:40 and 0:00:50\"\n}</tool_call>\n\n

# \n\n<tool_call>{\n  \"tool_name\": \"spatial_tracking\",\n  \"objects\": \[\"crates\", \"bags\"\]\n}</tool_call>\n\n
            

                

