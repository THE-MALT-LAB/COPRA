import os


save_folder = "./scores/"
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
from my_pipeline import PGPipeline

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

from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging



from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from my_vision_process import my_process_vision_info

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
                video_name = sample['video'].split('/')[-1]
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
        self.pipeline = PGPipeline(
            device_map="cuda",
        )

    def extract_answer(self, text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def forward(self, video_path: str, start_frame: int, end_frame: int, sampled_frame_idx: list[int], batch_idx=0):
        """ sample = pipeline.generate(
            prompt=prompt, imgs_input=imgs_input, diff_infer_steps=50, seed=42, verbose=2, image_size="1024x1024"
        ) """

        inputs = {}
        inputs['video_path'] = video_path
        messages = [{"role": "user", "content": [{"type": "video", "video": video_path, "total_pixels": 8192 * 28 * 28}]}]
        image_inputs, video_inputs, video_kwargs = my_process_vision_info(
            [messages],
            [(end_frame-start_frame)/30.], # TODO: Change fps hardcoded
            return_video_kwargs=True, 
            client=None,
            return_gt=True,
            my_frame_indices=[sampled_frame_idx]
        )
        fps_inputs = video_kwargs['fps']
        # # data_to_return["image_inputs"] = [torch.load(os.path.join(example["video_path"][0], "image_inputs.pt"))]
        inputs["video_inputs"] = [video_inputs]
        # with open(os.path.join(example["video_path"][0], "video_kwargs.json"), 'r') as f:
        inputs["video_kwargs"] = [video_kwargs]
        inputs["problem"] = f"There are {video_inputs[0].size(0)} input frames."
        inputs["solution"] = video_kwargs["gt"]

        responses = self.pipeline.generate( [inputs] )
        response = responses[0]
        predict_labels = 1
        answer = self.extract_answer( response )
        if answer == '0':
            predict_labels = 0
        elif answer == '1':
            predict_labels = 1
        #breakpoint()
        return predict_labels

# Create the datasets
test_dataset = Video_Instruct_Dataset_Inference(vis_root='ucf_crime/videos/', ann_root='ucf_crime/annotations/UCF_eval_vera.json')

# Create the data loaders
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

# Initialize Lightning Model
lightning_model = VideoAnomalyDetectionModelInference().eval().cuda()

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

        for ind in indices:

            snippet_start = max(ind - 1/2*fps*t_window, start)
            snippet_end = min(ind+1/2*fps*t_window, end)

            snippet_indices = np.arange(snippet_start, snippet_end, (snippet_end-snippet_start)/snippet_len).astype(int).tolist()
                
            predict_labels = lightning_model.forward(video_path[0], snippet_start, snippet_end, snippet_indices)

            dict_single[ind]={'start':snippet_start,'end':snippet_end, 'score':predict_labels}

        with open(save_folder+"{}.json".format(video_name[0]),"w") as outfile:
            
            json.dump(dict_single, outfile)

    except (StopIteration):
        break
 