import tqdm
import os

save_folder = "./scores_vru_accident_densecap/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

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
sys.path.append('/home/muye/project/InternVL/internvl_chat')

#from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging



from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Change prompt to VRU Accident specific for dense captioning
MY_SYSTEM_PROMPT = """You are the model.
** Model Description: **
You are designed to do Dense Captioning. The input is a sequence of video frames. Please answer the prompt questions.
** Prompt Questions: **
Provide a detailed description of this accident video. Use clear and complete sentences with appropriate traffic and accident-related terminology.
Include descriptions of weather conditions, road type, and vehicel or pedestrian appearance (such as clothing and posture).
Mention vehicle speed, trajectory, and movements, as well as any changes in the pedestrian's behavior. Focus on the dynamics of the collision, including vehicle approach, pedestrian movement, and final impact.
Please give your output strictly in the following format:
```
<answer>[Add your response here]</answer>
```
Please ONLY reply according to this format."""

class Video_Instruct_Dataset_Inference(Dataset):
    def __init__(self, vis_root, annotation_files):
        self.annotation = {}
        for ann_root in annotation_files:
            data_path = pathlib.Path(ann_root)
            with data_path.open(encoding='utf-8') as f:
                current_annotation = json.load(f)
                self.annotation = self.annotation | current_annotation
        self.example_names = list( self.annotation.keys() )
        #breakpoint()
        self.vis_root = vis_root
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        num_retries = 10  
        #breakpoint()
        for _ in range(num_retries):
            try:
                #breakpoint()
                video_name = self.example_names[ index ]
                sample = self.annotation[ video_name ]
                video_path = os.path.join(self.vis_root, video_name)
                #breakpoint()
            except Exception as e:
                print(f"Failed to load examples with video: {self.example_names[ index ]}. Will randomly sample an example as a replacement: {e}")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        return video_path, video_name, sample

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
        return text
    
    def make_conversation_video(self):
        # add support for options
        return {
            'prompt': [{
                'role': 'user',
                'content': [
                    *({'type': 'image', 'text': None} for _ in range(8)), # TODO: Hardcoded
                    {'type': 'text', 'text': MY_SYSTEM_PROMPT}
                ]
            }]
        }

    def inject_parameters(self, video_path: str):
        # Format into conversation

        # Load video inputs here
        #breakpoint()
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
        duration = (end_frame - start_frame) / 30.
        sample_fps = max_num_frames / duration
        #breakpoint()
        # Convert to PIL Image
        video = [
            Image.fromarray(frames[i]).resize(
                (320, 240),   # width, height for PIL
                resample=Image.BICUBIC
            )
            for i in range(len(frame_idx))
        ]
        
        #breakpoint()        
        filename = video_path.split('/')[-1].lower()
        solution = 0 if 'normal' in filename else 1
        prompt = self.make_conversation_video()['prompt']
        #breakpoint()
        trainer_inputs = {
            'video_path': video_path,
            'problem': MY_SYSTEM_PROMPT,
            'solution': solution,
            'prompt': prompt,
            'video': video
        }
        #breakpoint()

        self.pipeline.inject_parameters( [trainer_inputs] )
        return video
    
    def clear_injected_parameters(self,):
        self.pipeline.clear_injected_parameters()

    # add support for options
    def forward(self, video_path: str, video: list[Image]):
        filename = video_path.split('/')[-1].lower()

        # Build args for batch vision processing
        prompt = self.make_conversation_video()['prompt']
        #breakpoint()

        # Build batch_inputs for pipeline.generate
        batch_inputs = []
        trainer_inputs = {
            'video_path': video_path,
            'problem': MY_SYSTEM_PROMPT,
            'solution': None,
            'prompt': prompt,
            'video': video
        }
        batch_inputs.append(trainer_inputs)
        #breakpoint()
        #breakpoint()
        """
        for i, info in enumerate(chunk):
            inputs = {}
            inputs['video_path'] = video_path
            inputs['video_inputs'] = [[video_inputs_list[i]]]
            inputs['video_kwargs'] = [{'fps': [video_kwargs['fps'][i]], 'gt': video_kwargs['gt']}]
            inputs['problem'] = f"There are {video_inputs_list[i].size(0)} input frames."
            inputs['solution'] = video_kwargs['gt'][i]
            batch_inputs.append(inputs) """

        responses = self.pipeline.generate(batch_inputs)
        #breakpoint()
        answer = self.extract_answer( responses[0] )
        #breakpoint()
        results = { 'video_name': video_path, 'answer': answer }
        #breakpoint()
        return results

# Create the datasets
test_dataset = Video_Instruct_Dataset_Inference(
    vis_root='vru_accident', annotation_files=[
        'vru_accident/annotations/CAP_DATA/CAP_DATA_Dense_Caption.json',
        'vru_accident/annotations/DADA_2000/DADA_2000_Dense_Caption.json',
        'vru_accident/annotations/DoTA/DoTA_Dense_Caption.json',
        'vru_accident/annotations/MANUAL_DATA/MANUAL_DATA_Dense_Caption.json'
    ]
)

# Create the data loaders
test_loader = accelerator.prepare(
    DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
)

# Initialize Lightning Model
lightning_model = VideoAnomalyDetectionModelInference().eval()

iter_test_loader = iter(test_loader)

fps = 30
t_window = 10
sampling_rate = 16
snippet_len = 8
 
dict_all = {}
count = -1

while True:

    dict_single ={}
    try:
        video_path, video_name, sample  = next(iter_test_loader)
        count+=1

        print("testing video "+video_name[0])
        
        # inject here
        # question and options are ignored here
        video = lightning_model.inject_parameters( video_path[0] )
        vru_gt = sample['gt'][0]
        #breakpoint()
        # no longer need segment level processing 

        all_snippet_info = []
        dict_single = lightning_model.forward(video_path[0], video)
        dict_single['gt_answer'] = vru_gt
        #breakpoint()

        # use accelerator index and save to jsonl
        with open(
            save_folder+"vru_accident_cuda:{}.jsonl".format(accelerator.local_process_index),
            "a") as outfile:
            
            json.dump(dict_single, outfile)
            print("", file=outfile)
        
        # clear params here
        #breakpoint()
        lightning_model.clear_injected_parameters()

    except (StopIteration):
        break
 