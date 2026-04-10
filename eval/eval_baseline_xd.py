import tqdm
import os

save_folder = "./scores_baseline_xd/"
#if not os.path.exists(save_folder):
#    os.makedirs(save_folder)

# VERA Questions might hint about task. Remove for clean evaluation
""" new_qs = "1. Are there any unusual movements or positions of the vehicles in the video frames?"+"\n"+ \
"2. Are there any inconsistencies in the actions of the vehicles in the video frames that suggest an anomaly?"+"\n"+ \
"3. Are there any unusual behaviors of the vehicles in the video frames?"+"\n"+ \
"4. Are there any inconsistencies in the weather conditions in the video frames that suggest an anomaly?"+"\n"+ \
"5. Are there any unusual objects or items in the video frames that are not in their usual positions?"+"\n" """


import sys
import re
 

import os
from my_baseline_pipeline import BaselinePipeline

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
You are designed to do binary classification. The input is a sequence of video frames for identifying whether there is an anomaly in the video; you need to output the class label, i.e., an integer in the set {0, 1}. 0 represents normal video, and 1 represents abnormal video. Please answer the prompt questions.
** Prompt Questions: **
Answer the following questions based on what you see from the video frames and provide an explanation in one sentence. 
Based on the analysis above, please conclude your answer to 'Is there any anomaly in the video?' in 'Yes, there is an anomaly' or 'No, there is no anomaly'.
Please give your output strictly in the following format:
```
<think>[Add your reasoning here]</think>
<answer>[ONLY the integer class label; make necessary assumptions if needed]</answer>
```
Please ONLY reply according to this format, don't give me any other words."""

def fetch_segments_batch(
    video_path,
    all_durations: list[float],
    all_frame_indices: list[list[int]],  # one list of frame indices per segment
) -> list[Image]:
    """
    Open the video once, read all needed frames in a single get_batch call,
    then slice into per-segment tensors.
    """
    vr = decord.VideoReader(video_path)
    nframes_total, height, width = len(vr), *vr[0].shape[:2]

    # Collect all unique frame indices across all segments, read once
    all_flat_indices = []
    #breakpoint()
    for seg_indices in all_frame_indices:
        all_flat_indices.extend(seg_indices)
    unique_indices = sorted(set(all_flat_indices))
    index_to_pos = {idx: pos for pos, idx in enumerate(unique_indices)}

    # Single disk read
    all_frames = vr.get_batch(unique_indices).asnumpy()  # (N_unique, H, W, C)

    # Slice and resize per segment
    results = []
    for seg_indices, duration in zip(all_frame_indices, all_durations):
        pos = [index_to_pos[i] for i in seg_indices]
        video = all_frames[pos]  # (T, C, H, W)
        num_sampled_frame = len(seg_indices)

        video = [Image.fromarray(video[i]) for i in range(len(video))]
        
        results.append(video)

    return results

class Video_Instruct_Dataset_Inference(Dataset):
    def __init__(self, vis_root, ann_root):
        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
        
        self.vis_root = vis_root
    
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        num_retries = 10  
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                video_name = sample['video'].split('/')[-1] + '.mp4'
                video_path = os.path.join(self.vis_root, video_name)
                if 'Normal' in video_path:
                    video_label_vad = 0
                else:
                    video_label_vad = 1
                    
                vlen = sample['length']

            except:
                print(f"Failed to load examples with video: {video_path}. Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        return video_path, video_name, vlen

class VideoAnomalyDetectionModelInference(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.pipeline = BaselinePipeline(
            device_map="cuda",
        )

    def extract_answer(self, text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        #breakpoint()
        if match:
            return match.group(1).strip()
        return ""
    
    def make_conversation_video(self):
        return {
            'prompt': [{
                'role': 'user',
                'content': [
                    *({'type': 'image', 'text': None} for _ in range(8)), # TODO: Hardcoded
                    {'type': 'text', 'text': MY_SYSTEM_PROMPT}
                ]
            }]
        }

    def forward(self, video_path: str, start_frame: int, end_frame: int,
                all_snippet_info: list[dict], batch_size: int = 8, batch_idx=0):
        filename = video_path.split('/')[-1].lower()
        solution = 0 if 'normal' in filename else 1
        results = {}
        for chunk_start in tqdm.tqdm(range(0, len(all_snippet_info), batch_size)):
            chunk = all_snippet_info[chunk_start:chunk_start + batch_size]

            # Build args for batch vision processing
            prompt = self.make_conversation_video()['prompt']
            durations = [(info['end'] - info['start']) / 30. for info in chunk]
            all_frame_indices = [info['snippet_indices'] for info in chunk]

            # Single file open, single get_batch — replaces the per-segment loop
            all_segments = fetch_segments_batch(
                video_path,
                durations,
                all_frame_indices,
            )

            # Build batch_inputs for pipeline.generate
            batch_inputs = []
            for i, info in enumerate(chunk):
                trainer_inputs = {
                    'video_path': video_path,
                    'problem': MY_SYSTEM_PROMPT,
                    'solution': solution,
                    'prompt': prompt,
                    'video': all_segments[i]
                }
                batch_inputs.append(trainer_inputs)
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
            for info, response in zip(chunk, responses):
                answer = self.extract_answer(response)
                predict_labels = 0 if answer == '0' else 1
                results[info['ind']] = {'start': info['start'], 'end': info['end'], 'score': predict_labels}
        #breakpoint()
        return results

# Create the datasets
test_dataset = Video_Instruct_Dataset_Inference(vis_root='xd_violence/videos/', ann_root='xd_violence/xd_test_800.json')

# Create the data loaders
test_loader = accelerator.prepare(
    DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
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
        video_path, video_name, n_frms  = next(iter_test_loader)
        count+=1

        print("testing video "+video_name[0])

        import os.path

        if os.path.exists(save_folder+"{}.json".format(video_name[0])):
            continue
        
        start, end = 0, n_frms.item()
        vlen = n_frms.item()
        
        indices = [i for i in range(start, end, sampling_rate)]

        all_snippet_info = []
        for ind in indices:
            snippet_start = max(ind - 1/2*fps*t_window, start)
            snippet_end = min(ind + 1/2*fps*t_window, end)
            snippet_indices = np.arange(snippet_start, snippet_end, (snippet_end-snippet_start)/snippet_len).astype(int).tolist()
            all_snippet_info.append({
                'ind': ind,
                'start': snippet_start,
                'end': snippet_end,
                'snippet_indices': snippet_indices
            })
        dict_single = lightning_model.forward(video_path[0], snippet_start, snippet_end, all_snippet_info, batch_size=24)
        #breakpoint()

        with open(save_folder+"{}.json".format(video_name[0]),"w") as outfile:
            
            json.dump(dict_single, outfile)
        
    except (StopIteration):
        break
 