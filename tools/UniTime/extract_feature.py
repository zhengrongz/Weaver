import torch
from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor
from qwen_vision_process import process_vision_info
from feature import feature
import json
from torch.utils.data import Dataset, DataLoader
import os
import datetime
from tqdm import *
import decord


def get_duration(video_dir):
    vr = decord.VideoReader(video_dir)

    fps = vr.get_avg_fps()
    num_frames = len(vr)

    duration = num_frames / fps

    return duration





class LongVideoReasonDataset(Dataset):
    def __init__(self, data_dir, video_dir):
        self.test_datas = []
        with open(data_dir, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 900:
                    continue
                # 解析每行的JSON数据
                # if i >= 900:
                #     break
                json_obj = json.loads(line.strip())
                self.test_datas.append(json_obj)
        self.video_dir = video_dir
        self.video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    
    def __len__(self):
        return len(self.test_datas)
    
    def __getitem__(self, idx):

        line = self.test_datas[idx]

        video_dir = os.path.join(self.video_dir, line['videos'])



        output = {
            'video_dir': video_dir,
            'record': line,
        }
        return output

def dict_collate_fn(batch):
    # import ipdb; ipdb.set_trace()
    batch_data = [item for item in batch]
    return batch_data



def main():
    device = torch.device("cuda")
    # setup(rank, world_size)
    
    # 初始化模型和处理器（每个进程独立）
    # model_id = "/mnt/ali-sh-1/usr/shiyudi1/REVPT/checkpoints/visual_process_r1/qwen2_5_vl_7b_mix_0/global_step_40/actor/huggingface"
    processor = Qwen2VLMRProcessor.from_pretrained("/mnt/ali-sh-1/dataset/zeus/shiyudi/models/Qwen2-VL-7B-Instruct")
    
    model = Qwen2VLMRForConditionalGeneration.from_pretrained(
        "/mnt/ali-sh-1/dataset/zeus/shiyudi/models/UniTime",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    # model = model.to(rank)
    # model = DDP(model, device_ids=[rank])
    model.eval()


    # dataset = LongVideoReasonDataset("/mnt/ali-sh-1/dataset/zeus/shiyudi/longvideoreason/test.jsonl", "/mnt/ali-sh-1/dataset/zeus/shiyudi/longvideoreason/longvideo_eval")
    # sampler = DistributedSampler(dataset, shuffle=False)  # 测试集通常不shuffle
    # dataloader = DataLoader(dataset, batch_size=1, collate_fn=dict_collate_fn, num_workers=4)
    datas = []
    data_dir = "/mnt/ali-sh-1/dataset/zeus/shiyudi/LVBench/test.jsonl"
    with open(data_dir, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # 解析每行的JSON数据
            # if i >= 900:
            #     break
            json_obj = json.loads(line.strip())
            datas.append(json_obj)
    num = 0
    for i, data in enumerate(tqdm(datas)):
    # for i, test_data in enumerate(tqdm(dataloader, desc=f"progress")):
        # if i < 21:
        #     continue  # 跳过前面的数据
            
        # test_data = test_data[0]
        
        # import ipdb; ipdb.set_trace()
        
        video_dir = os.path.join("/mnt/ali-sh-1/dataset/zeus/liuyikun/dataset/LVBench/all_videos", f"{data['key']}.mp4")

        duration = get_duration(video_dir)

        if duration > 256:
            feature_path = feature(model, processor, video_dir, feature_root="/mnt/ali-sh-1/dataset/zeus/shiyudi/tool_rl/tmp_features")


if __name__ == "__main__":
    main()
    # world_size = 8
    # torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)