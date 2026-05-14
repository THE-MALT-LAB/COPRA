import tqdm
import os
import PIL
from PIL import Image

# VERA Questions might hint about task. Remove for clean evaluation
""" new_qs = "1. Are there any unusual movements or positions of the vehicles in the video frames?"+"\n"+ \
"2. Are there any inconsistencies in the actions of the vehicles in the video frames that suggest an anomaly?"+"\n"+ \
"3. Are there any unusual behaviors of the vehicles in the video frames?"+"\n"+ \
"4. Are there any inconsistencies in the weather conditions in the video frames that suggest an anomaly?"+"\n"+ \
"5. Are there any unusual objects or items in the video frames that are not in their usual positions?"+"\n" """


import sys
import re
 

import os
from my_vid_level_pg_pipeline import PGPipeline

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=gpu_index


from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import decord
from decord import VideoReader, cpu
import random
import torch
from torch.utils.data.dataloader import default_collate

from accelerate import Accelerator
accelerator = Accelerator()
# Tell CUDA which GPU this process should use
torch.cuda.set_device(accelerator.local_process_index)

my_save_file = f"hivau_output_cuda:{accelerator.local_process_index}.jsonl"

from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
import math
from torchvision import transforms
import pdb
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel
import pytorch_lightning as pl
import itertools

import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers

import sys
#sys.path.append('/home/muye/project/InternVL/internvl_chat')

#from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging



from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

MY_SYSTEM_PROMPT = """You are the model.
** Model Description: **
You are designed to do video anomaly understanding. The input is a sequence of video frames for understanding whether there is an anomaly in the video or not. Please answer the prompt question.
** Prompt Question: **
Answer the following question based on what you see from the video frames and provide an explanation in one sentence. 
    {}
Please give your output strictly in the following format:
```
<think>[Add your reasoning here]</think>
<answer>[Add your response here]</answer>
```
Please ONLY reply according to this format, don't give me any other words."""

class Video_Instruct_Dataset_Inference(Dataset):
    def __init__(self, vis_root, ann_root):
        data_path = pathlib.Path(ann_root)
        # Accomodate JSONL
        self.annotation = []
        with data_path.open(encoding='utf-8') as f:
            for line in f.readlines():
                self.annotation.append( json.loads( line.strip() ) )
        #breakpoint()
        self.vis_root = vis_root
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        #breakpoint()
        sample = self.annotation[index]
        # Change getter for video reading to match HIVAU
        video_name = sample['video']
        video_path = os.path.join(self.vis_root, video_name)
        #breakpoint()
        problem_id = sample['id']
        video_type = sample['type']
        task = sample['task']
        problem = sample['prompt']
        solution = sample['anwser'] # typo in hivau
        #breakpoint()
        #vlen = sample['length']
        return video_path, problem, solution, video_type, problem_id

class VideoAnomalyDetectionModelInference(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.pipeline = PGPipeline(
            device_map="cuda",
        )

    def extract_answer(self, text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        #breakpoint()
        if match:
            return match.group(1).strip()
        return ""
    
    def make_conversation_video(self, problem):
        return {
            'prompt': [{
                'role': 'user',
                'content': [
                    *({'type': 'image', 'text': None} for _ in range(8)), # TODO: Hardcoded
                    {'type': 'text', 'text': MY_SYSTEM_PROMPT.format(problem)}
                ]
            }]
        }
    
    def clear_injected_parameters(self,):
        self.pipeline.clear_injected_parameters()

    def forward(self, video: list[PIL.Image], problem: str,
                solution: str):
        results = {}
        # Build args for batch vision processing
        prompt = self.make_conversation_video(problem)['prompt']

        # Build batch_inputs for pipeline.generate
        batch_inputs = []
        trainer_inputs = {
            'video_path': video_path,
            'problem': MY_SYSTEM_PROMPT.format(problem),
            'solution': solution,
            'prompt': prompt,
            'video': video
        }
        batch_inputs.append(trainer_inputs)
        #breakpoint()
        self.pipeline.inject_parameters( batch_inputs )
        #breakpoint()
        responses = self.pipeline.generate(batch_inputs)
        #breakpoint()
        answer = self.extract_answer(responses[0])
        #breakpoint()
        return answer

# Create the datasets
test_dataset = Video_Instruct_Dataset_Inference(vis_root='HIVAU-70k/videos/', ann_root='HIVAU-70k/instruction/my_merge_instruction_test_final.jsonl')

# Create the data loaders
test_loader = accelerator.prepare(
    DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4) 
)

# Initialize Lightning Model
lightning_model = VideoAnomalyDetectionModelInference().eval()

iter_test_loader = iter(test_loader)
print(len(test_loader), f"on cuda:{accelerator.local_process_index}")

snippet_len = 8
 
count = -1

 
while True:

    dict_single ={}
    try:
        video_path, problem, \
            solution, video_type, \
            problem_id = next(iter_test_loader)
        video_path = video_path[0]
        problem = problem[0]
        solution = solution[0]
        video_type = video_type[0]
        problem_id = problem_id[0].item()
        count+=1
        #breakpoint()
        print("testing video "+video_path)
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        start_frame = 0
        end_frame = total_frames
        # Sample 8 evenly spaced frame indices using VERA code
        # TODO: Hardcoded
        max_num_frames = min(8, total_frames)
        random_integers = np.arange(start_frame, end_frame, (end_frame - start_frame) / max_num_frames).astype(int).tolist() 
        sorted_integers = np.sort(random_integers)
        frame_idx = sorted_integers.astype(int)
        frames = vr.get_batch(frame_idx).asnumpy()  # (8, H, W, 3)
        #duration = (end_frame - start_frame) / 30.
        #sample_fps = max_num_frames / duration
        #breakpoint()
        # Convert to PIL Image
        video = [Image.fromarray(frames[i]) for i in range(len(frame_idx))]
        #breakpoint()
        answer = lightning_model.forward(video, problem, solution)
        #breakpoint()
        
        dict_single = {
            'id': problem_id,
            'type': video_type,
            'video': video_path,
            'prompt': problem,
            'answer': solution,
            'my_response': answer
        }

        with open(my_save_file,"a") as outfile:
            json.dump(dict_single, outfile)
            print("\n", end='', file=outfile)
        #breakpoint()
        lightning_model.clear_injected_parameters()

    except (StopIteration):
        break
 