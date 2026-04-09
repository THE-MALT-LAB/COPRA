# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from grpo_trainer import MyGRPOTrainer
from open_r1.trainer import GRPOConfig
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
monkey_patch_qwen2_5vl_flash_attn()
import decord
import numpy as np

# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    num_sampled_frames: Optional[str] = field(
        default=8,
        metadata={"help": "Maximum number of sampled images from video path"}
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

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

class WeaklySupervisedVideoAnomalyDetectionDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(WeaklySupervisedVideoAnomalyDetectionDataset, self).__init__()
        self.script_args = script_args
        self.data_root = self.script_args.image_root
        self.list_data_dict = []
        self.question_template = f"There are {self.script_args.num_sampled_frames} input frames."
        
        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    random.shuffle(cur_data_dict)
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation_video(example):
            return {
                'prompt': [{
                    'role': 'user',
                    'content': [
                        *({'type': 'image', 'text': None} for _ in range(self.script_args.num_sampled_frames)),
                        {'type': 'text', 'text': MY_SYSTEM_PROMPT}
                    ]
                }]
            }

        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        
        video_path = os.path.join(self.data_root, example['video'])
        # Load video inputs here
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        # Sample 8 evenly spaced frame indices using VERA code
        start_frame = 0
        end_frame = total_frames
        max_num_frames = min(self.script_args.num_sampled_frames, total_frames)
        random_integers = np.arange(start_frame, end_frame, (end_frame - start_frame) / max_num_frames).astype(int).tolist() 
        sorted_integers = np.sort(random_integers)
        frame_idx = sorted_integers.astype(int)
        frames = vr.get_batch(frame_idx).asnumpy()  # (8, H, W, 3)
        # TODO: insert into inputs
        sample_fps = max_num_frames / example['duration']
        #breakpoint()
        """ start, end = 0, vlen
        # Uniform sampling
        seg_indices = np.arange(start, end, vlen / self.num_seg).astype(int).tolist()
        seg_indices = seg_indices + [end]
        indices = []
        for _ in range(len(seg_indices) - 1):
            seg_begin = seg_indices[_]
            seg_end = seg_indices[_ + 1]
            if self.test_flag:
                random_integers = np.arange(seg_begin, seg_end, (seg_end - seg_begin) / self.num_frm_seg).astype(int).tolist()
            else:
                random_integers = np.random.choice(np.arange(seg_begin, seg_end), size=self.num_frm_seg, replace=False)
            sorted_integers = np.sort(random_integers)
            indices += sorted_integers.tolist() """
        #breakpoint()
        # Convert to PIL Image
        video = [Image.fromarray(frames[i]) for i in range(len(frame_idx))]
        #breakpoint()
        category = example['video_type'].lower()
        solution = 0 if 'normal' == category else 1
        prompt = make_conversation_video(example)['prompt']
        #breakpoint()
        trainer_inputs = {
            'video_path': video_path,
            'problem': MY_SYSTEM_PROMPT,
            'solution': solution,
            'prompt': prompt,
            'video': video
        }
        #breakpoint()
        return trainer_inputs


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    elif "glm" in model_name_or_path.lower():
        return GLMVModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def my_accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    #breakpoint()
    contents = [completion[0]['content'] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
    
        try:
            output_ans = extract_answer(content)
            gt_ans = str( sol )
            #breakpoint()
            reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    #breakpoint()
    return rewards


def my_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]['content'] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    #breakpoint()
    return [1.0 if match else 0.0 for match in matches]

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)

    # Load the reward functions
    reward_funcs_registry = {
        "accuracy": my_accuracy_reward,
        "format": my_format_reward,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = WeaklySupervisedVideoAnomalyDetectionDataset(script_args.dataset_name, script_args)

    trainer_cls = MyGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
        num_sampled_frames=script_args.num_sampled_frames
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
