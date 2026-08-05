import csv
import codecs

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch
import decord
from qwen_vl_utils import process_vision_info
import json
from tqdm import *
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import math
import numpy as np
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
from torch.utils.data import Dataset, DataLoader
import re
import random
import pyarrow.parquet as pq
import pysubs2
import ast
import datetime
from verl.workers.agent.qwen_tools import fetch_tools
import cv2
import time


def save_video(video, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    if video.shape[-1] != 3:
        saved_video = np.transpose(video, (0, 2, 3, 1))
    else:
        saved_video = video
    if type(saved_video) == torch.Tensor:
        saved_video = saved_video.cpu().numpy()
    T, H, W, C = saved_video.shape

    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for i in range(T):
        frame = saved_video[i]
        
        if C == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif C == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        if frame.dtype != np.uint8:
            frame = (frame).astype(np.uint8)
        
        out.write(frame)
    
    out.release()

def get_duration(video_dir):
    vr = decord.VideoReader(video_dir)

    fps = vr.get_avg_fps()
    num_frames = len(vr)

    duration = num_frames / fps

    return duration


def generate(model, processor, messages, rank):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(rank)

    with torch.no_grad():
        generated_ids = model.module.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
    
    return output_text, video_inputs

def process_and_action(video, video_dir, duration, output_text, tools):
    video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
    obs_template = "<|im_end|>\n<|im_start|>user\n{vision}\n{text}.<|im_end|>\n<|im_start|>assistant\n"
    answer_pattern = r"<answer>(.*?)</answer>"
    tool_use_pattern = r"<tool_call>(.*?)</tool_call>"
    if isinstance(video[-1], torch.Tensor):
        env_state = {
            "video": video[-1].cpu().numpy()
        }
    else:
        env_state = {
            "video": video[-1]
        }
    data_args = {
        "video_path": video_dir,
        "duration": duration
    }

    answer_match = re.search(answer_pattern, output_text, re.DOTALL)

    if answer_match:
        pred_answer = answer_match.group(1).strip()
    else:
        pred_answer = None
    
    tool_match = re.search(tool_use_pattern, output_text, re.DOTALL)
    if tool_match:
        tool_dict = tool_match.group(1).strip()

        try:
            tool_dict = ast.literal_eval(tool_dict)
        
            tool_name = tool_dict.get("tool_name")
            print(tool_name)
            query = tool_dict.get("query", None)
            objects = tool_dict.get("objects", None)
            start = tool_dict.get("start", None)
            end = tool_dict.get("end", None)
            if query:
                data_args['query'] = query
            if objects:
                data_args['objects'] = objects
            if start:
                data_args['start'] = start
            if end:
                data_args['end'] = end
            # import ipdb; ipdb.set_trace()
            start = time.time()
            tool_result = tools[tool_name].call(data_args, env_state)
            end = time.time()
            print(end - start)

            tool_obs_text = tool_result.get('text', '')
            tool_obs_video = tool_result.get("video", None)
            # import ipdb; ipdb.set_trace()

            final_obs_content = obs_template.format(vision=video_placeholder, text=tool_obs_text)

            return final_obs_content, tool_obs_video, pred_answer
        except:
            final_obs_content = obs_template.format(vision='', text='Using tool failed, rethink to select use tool or answer.')
            return final_obs_content, None, pred_answer
    else:
        final_obs_content = obs_template.format(vision='', text='Without tool usage, continue analyzing.')
        return final_obs_content, None, pred_answer


def load_parquet(parquet_file):
    table = pq.read_table(parquet_file)

    # Convert PyArrow Table to pandas DataFrame
    df = table.to_pandas()

    jsons = []
    for record in df.itertuples():

        if len(jsons) < int(record.video_id):
            jsons.append({
                "video_id": record.video_id,
                "youtube_id": record.videoID,
                "url": record.url,
                "duration": record.duration,
                "domain": record.domain,
                "sub_category": record.sub_category,
                "questions": [
                    {
                        "question_id": record.question_id,
                        "task_type": record.task_type,
                        "question": record.question,
                        "choices": list(record.options),
                        "answer": record.answer,
                    }
                ]
            })
        else:
            jsons[-1]['questions'].append({
                "question_id": record.question_id,
                "task_type": record.task_type,
                "question": record.question,
                "choices": list(record.options),
                "answer": record.answer,
            })

    return jsons



def calculate_metrics(gts, preds):
    num_acc = 0
    nums = len(gts)
    for i in range(len(gts)):
        gt = gts[i]
        pred = preds[i]

        if gt == pred:
            num_acc += 1
    
    acc = num_acc / nums

    return {"Acc": acc}


def transfer_answer(result, answer_num):
    bracket_match = re.match(r'\(([A-Z])\)', result[0])
    if bracket_match:
        return bracket_match.group(1)
    
    if result[0] and result[0][0].isalpha() and result[0][0].isupper():
        return result[0][0]
    
    option_match = re.search(r'[A-Z]', result[0])
    if option_match:
        return option_match.group(0)
    
    return chr(ord('A') + random.randint(0, min(answer_num - 1, 25)))  # 确保不超过Z



class LongVideoReasonDataset(Dataset):
    def __init__(self, data_dir, video_dir, num_samples=None):
        self.test_datas = []
        with open(data_dir, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line.strip())
                self.test_datas.append(json_obj)
        if num_samples is not None and num_samples > 0:
            self.test_datas = random.sample(self.test_datas, min(num_samples, len(self.test_datas)))
        self.video_dir = video_dir
        self.video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    
    def __len__(self):
        return len(self.test_datas)
    
    def __getitem__(self, idx):

        line = self.test_datas[idx]

        video_dir = os.path.join(self.video_dir, line['videos'])
        base_name = os.path.basename(video_dir).split(".mp4")[0]
        frame_dir = os.path.join(self.video_dir, "frames", base_name)



        output = {
            'video_dir': video_dir,
            "frame_dir": frame_dir,
            'record': line,
        }
        return output


def dict_collate_fn(batch):
    # import ipdb; ipdb.set_trace()
    batch_data = [item for item in batch]
    return batch_data

def setup(rank, world_size):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    timeout = datetime.timedelta(hours=4)
    dist.init_process_group("nccl", rank=rank, timeout=timeout, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="Interleave eval for LongVideoReason benchmark")
    # Paths
    parser.add_argument("--model_id", type=str, required=True,
                        help="Path to the model checkpoint (HuggingFace format)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the test JSONL file")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Root directory containing video files")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Directory to save prediction results (default: eval_results)")
    # Dataset
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to randomly evaluate (default: all)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num_workers (default: 4)")
    # Model inference
    parser.add_argument("--nframes", type=int, default=128,
                        help="Number of frames to sample per video (default: 128)")
    parser.add_argument("--max_pixels", type=int, default=448*448,
                        help="Max pixels per frame (default: 448*448=200704)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens per generation step (default: 512)")
    parser.add_argument("--max_turns", type=int, default=10,
                        help="Max tool-call turns per sample (default: 10)")
    # Distributed
    parser.add_argument("--world_size", type=int, default=4,
                        help="Number of GPUs to use (default: 4)")
    return parser.parse_args()


def main(rank, world_size, args):
    setup(rank, world_size)
    
    model_id = args.model_id
    processor = AutoProcessor.from_pretrained(model_id)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    model.eval()

    tools = fetch_tools()

    dataset = LongVideoReasonDataset(args.data_dir, args.video_dir, num_samples=args.num_samples)
    sampler = DistributedSampler(dataset, shuffle=False)  # 测试集通常不shuffle
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=dict_collate_fn, num_workers=args.num_workers)

    system_prompt = """
    You are a helpful multimodal assistant. Your task is to solve complex visual questions by thinking step-by-step and using tools.

    #Tools
    You are provided with following tools:
    1. Temporal grounding tool: Grounds a detailed temporal window of a certain event according to the "query". Usage: <tool_call>{"tool_name": "temporal_grounding", "query": "the event you want to ground."}</tool_call>
    2. Spatial tracking tool: Tracks certain objects in spatial bounding boxes. Usage: <tool_call>{"tool_name": "spatial_tracking", "objects": ["object1", "object2", ...]}</tool_call>
    3. Frame selection tool: Selects a single, most representative keyframe that best matches a textual query. This is ideal for answering questions about static scenes that do not require temporal analysis, such as counting objects or identifying attributes. Usage: <tool_call>{"tool_name": "frame_selection", "query": "a textual description of the desired scene or moment."}</tool_call>
    4. Spatial grounding tool: Locates specified objects with bounding boxes in a more specific static scene. This tool operates on the current focused context, which should be a static scene (ideally a single frame) prepared by a preceding tool like frame_selection or a very short trim. If the context contains multiple frames, it will default to analyzing the middle frame. Usage: <tool_call>{"tool_name": "spatial_grounding", "objects": ["object1", "object2", ...]}</tool_call>
    5. Trim tool: Grounds a detailed temporal window if you can get direct start and end timestamps from question, options or thinking process. It is also useful when you want to look at the video clip which is "before" or "after" some events, the default "start" is 0 and "end" is the duration of the video if you don't provide. The timestamp should be transfered into seconds. Usage: <tool_call>{"tool_name": "trim", "start": start timestamp, should be a float, "end": end timestamp, should be a float.}</tool_call>
    6. Temporal Count tool: Ground a detailed event which can happen multiple times in the video, the tool will ground all related clip and concat them together. It should be used when the question explicitly requires count the number of some events. Usage: <tool_call>{"tool_name": "temporal_count", "query": "the event you want to count."}</tool_call>

    #Instructions
    1. In each step, you need to give a decomposed thinking process, and evaluate whether it is needed to use tools and which tools to use.
    2. You need to consider carefully which tool to use for similar type for different question type. (temporal grounding, temporal count and frame selection) (spatial tracking and spatial grounding)
    3. After calling tools and getting return results, you need to analyze the results and judge whether the results is useful. If not, you can recall the tool with different parameters.
    4. The results obtained from the tool may not always be accurate. You need to carefully watch the newly obtained fragments and analyze whether there is content you need. If not, proceed to the next step of analysis.
    5. If you think the process is ended and no more steps in needed, you need to output the final answer in <answer></answer> tags. e.g. <answer>Answer here</answer>.
    """



    gt_answers = []
    pred_answers = []
    records = []

    token_lens = [0]
    for i, test_data in enumerate(tqdm(dataloader, desc=f"Rank {rank}")):
            
        test_data = test_data[0]
        
        # import ipdb; ipdb.set_trace()
        
        video_dir = test_data['video_dir']
        frame_dir = test_data['frame_dir']
        record = test_data['record']
        question = record['problem']
        answer = record['answer']
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, answer, re.DOTALL)
        gt_answer = answer_match.group(1).strip()

        duration = get_duration(video_dir)
        
        instruct = question
        instruct += "Think step by step and call tools if needed, then answer."



        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_dir, "nframes": args.nframes, "max_pixels": args.max_pixels}, #"video_start": temporal_window[0], "video_end": temporal_window[1]},
                    {"type": "text", "text": instruct}
                ]
            },
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(rank)

        max_turns = args.max_turns

        turn = 0
        response = ''
        
        while turn <= max_turns:
            with torch.no_grad():
                try:
                    generated_ids = model.module.generate(**inputs, max_new_tokens=args.max_new_tokens)
                except Exception as e:
                    print(e)
                    print(tool_obs_text)
                    print(video_dir)
                    print(instruct)
                    import ipdb; ipdb.set_trace()
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            tool_obs_text, tool_obs_video, pred_answer = process_and_action(video_inputs, video_dir, duration, output_text, tools)
            if tool_obs_video is not None:
                video_inputs.append(tool_obs_video)

            response += output_text
            response += tool_obs_text

            if pred_answer:
                token_len = generated_ids[0].shape[0]
                token_lens[0] += token_len
                break

            new_tokens = torch.tensor(processor.tokenizer.encode(output_text), device=rank).unsqueeze(0)
            new_attention_mask = torch.ones([1, new_tokens.shape[1]], device=rank)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], new_tokens], dim=1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], new_attention_mask], dim=1)


            new_inputs = processor(
                text=[tool_obs_text],
                images=None,
                videos=tool_obs_video,
                return_tensors="pt",
            ).to(rank)

            obs_inputs = new_inputs['input_ids']

            inputs['input_ids'] = torch.cat([inputs['input_ids'], obs_inputs], dim=1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], new_inputs['attention_mask']], dim=1)
            if 'pixel_values_videos' in new_inputs:
                inputs['pixel_values_videos'] = torch.cat([inputs['pixel_values_videos'], new_inputs['pixel_values_videos']], dim=0)
            if 'video_grid_thw' in new_inputs:
                inputs['video_grid_thw'] = torch.cat([inputs['video_grid_thw'], new_inputs['video_grid_thw']], dim=0)
            if 'second_per_grid_ts' in new_inputs:
                inputs['second_per_grid_ts'].extend(new_inputs['second_per_grid_ts'])


            # import ipdb; ipdb.set_trace()

            turn += 1


            




        if not pred_answer:
            pred_answer = 'A'
        
        # import ipdb; ipdb.set_trace()
        pred = transfer_answer(pred_answer, 3)
        pred_answers.append(pred)
        gt_answers.append(gt_answer)


        record['pred'] = pred
        record['thinking'] = response
        records.append(record)
            

 

            


        

    dist.barrier()
    all_gt_answers = [None] * world_size
    all_pred_answers = [None] * world_size
    all_records = [None] * world_size
    all_token_lens = [None] * world_size
    dist.all_gather_object(all_gt_answers, gt_answers)
    dist.all_gather_object(all_pred_answers, pred_answers)
    dist.all_gather_object(all_records, records)
    dist.all_gather_object(all_token_lens, token_lens)

    if rank == 0:
        final_gt_answers = []
        final_pred_answers = []
        final_records = []
        final_token_lens = []
        for i in range(world_size):
            final_gt_answers.extend(all_gt_answers[i])
            final_pred_answers.extend(all_pred_answers[i])
            final_records.extend(all_records[i])
            final_token_lens.extend(all_token_lens[i])
        
        metrics = calculate_metrics(final_gt_answers, final_pred_answers)
        print(metrics)

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "predictions.json")
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(final_records, f, ensure_ascii=False, indent=2)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {results_path}")
        print(f"Metrics saved to {metrics_path}")

    cleanup()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(main, args=(args.world_size, args), nprocs=args.world_size)