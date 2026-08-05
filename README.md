# Weaver
Official PyTorch code of "Weaver: End-to-End Agentic System Training for Video Interleaved Reasoning", CVPR 2026.

[[Project page]](https://zhengrongz.github.io/Weaver/) [[Paper]](https://arxiv.org/abs/2602.05829)
[[Data]](https://huggingface.co/datasets/Zhengrongzz/Weaver)


## 🔥News
* **[2026.8.5]** We released code and data of Weaver!
* **[2026.4.23]** Weaver has been accepted to **CVPR 2026 Findings**!
* **[2026.2.6]** Weaver is released to Arxiv! Code is in progress, please stay tuned!


## Installation

Weaver uses two separate conda environments: **`weaver`** (main training environment) and **`tools`** (tool server environment).

### 1. Weaver Environment

The `weaver` environment is used for SFT and RL training.

```bash
# One-shot setup (creates conda env, installs GPU packages, and installs local packages)
bash setup_env.sh

# Activate the environment
conda activate weaver
```

`setup_env.sh` performs the following steps:
1. Creates the conda environment from `environment.yml` (Python 3.10, CPU packages)
2. Installs GPU packages (torch, flash-attn, vllm, etc.) via `install_gpu_packages.sh`
3. Installs local editable packages: `verl` and `qwen-vl-utils`

### 2. Tools Environment

The `tools` environment is used to run the tool servers (Temporal, Tracking, Spatial Grounding, etc.) that the agent calls during RL rollout.

Please refer to the individual tool READMEs for installation instructions:
- [`tools/UniTime/README.md`](tools/UniTime/README.md) — Temporal grounding tool
- [`tools/GroundedSAM2/README.md`](tools/GroundedSAM2/README.md) — Object tracking tool

Please download all the needed tool checkpoints and put it at the right place!

After setting up the `tools` environment, launch all tool servers before RL training:

```bash
# Launch tool servers for RL training
conda activate tools
python tools/lanuch_tools.py --config tools/train_tools_config.json
```

The config file [`tools/train_tools_config.json`](tools/train_tools_config.json) defines each tool's conda env, GPU assignment, port, and required environment variables. Edit the `env` fields to point to your model checkpoints and data paths before launching.


## Training

### Stage 0: Data
Please download the data first and put it at the right place and modify the corresponding scripts to load the data.
It is noted that the raw videos need to be downloaded from the raw project pages.

### Stage 1: SFT

SFT is performed using the `qwen-vl-finetune` submodule with the `weaver` conda environment.

```bash
conda activate weaver
cd qwen-vl-finetune

# Edit scripts/sft.sh to set your model path, dataset, and output directory
bash scripts/tool_sft.sh
```

### Stage 2: RL Training

RL training uses GRPO with tool-augmented rollout. Before launching, make sure the tool servers are running (see [Tools Environment](#2-tools-environment) above).

```bash
conda activate weaver

# Edit scripts/video_run.sh to set CKPT_PATH, TRAIN_FILES, VAL_FILES
# Noted that if you want to deploy tools on other nodes, you need to modify the url in verl/trainer/config/ppo_trainer.yaml, more detailed configs can also be modified here.
bash scripts/video_run.sh
```



## Evaluation

Please refer to the [`eval/`](eval/) folder for evaluation scripts and instructions.

As an example, to evaluate on the LVReason benchmark:

```bash
conda activate weaver

# Set MODEL_ID, DATA_DIR, VIDEO_DIR, OUTPUT_DIR as needed
MODEL_ID=/path/to/checkpoint \
DATA_DIR=/path/to/test.jsonl \
VIDEO_DIR=/path/to/videos \
OUTPUT_DIR=eval_results/interleave \
bash eval/LVReason/eval_interleave.sh
```

## Acknowledgements
Thanks to several excellent open-source projects:

* [REVPT](https://github.com/ls-kelvin/REVPT)
* [Deepeyes](https://github.com/Visual-Agent/DeepEyes)
* [Video-R1](https://github.com/tulerfeng/Video-R1)
* [Longvideo-Reason](https://github.com/NVLabs/Long-RL) 



## Citation
If you find this paper useful, please consider staring this repo and citing our paper!
```latex
@article{shi2026weaver,
  title={Weaver: End-to-End Agentic System Training for Video Interleaved Reasoning},
  author={Shi, Yudi and Di, Shangzhe and Chen, Qirui and Wang, Qinian and Cai, Jiayin and Jiang, Xiaolong and Hu, Yao and Xie, Weidi},
  journal={arXiv preprint arXiv:2602.05829},
  year={2026}
}
```
