from typing import Optional, Union, Any
from collections.abc import Iterator, Mapping

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig

from wu.config import ParameterGeneratorConfig
from wu.generator import ParameterGenerator
from wu.mixin import inject_pg


class PGPipeline:
    def __init__(
        self,
        device_map: str = "auto",
    ):
        self.model_id = "Qwen/Qwen2-VL-7B-Instruct"
        self.SYSTEM_PROMPT =  """You are the model.
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

        # TODO: Make this work for more backbones
        self.parameter_generator_config = ParameterGeneratorConfig(
            token_dim=512,
            rank=8, # 16 OOM
            alpha=16, # 2x rank
            input_dim=3584,
            dim_accumulation=4,
            d_model=512,
            head_dim=512,
            num_base_model_layers=28,
            output_dim=3584,
            num_pg_layers=14,
            prefix="model.language_model.layers.",
            pg_mapping={
                "self_attn.q_proj": {"lora_A_dim": 3584, "lora_B_dim": 3584},
                "self_attn.k_proj": {"lora_A_dim": 3584, "lora_B_dim": 512},
                "self_attn.v_proj": {"lora_A_dim": 3584, "lora_B_dim": 512},
                "self_attn.o_proj": {"lora_A_dim": 3584, "lora_B_dim": 3584},
                "mlp.gate_proj": {"lora_A_dim": 3584, "lora_B_dim": 18944},
                "mlp.up_proj": {"lora_A_dim": 3584, "lora_B_dim": 18944},
                "mlp.down_proj": {"lora_A_dim": 18944, "lora_B_dim": 3584},
            },
            torch_dtype=torch.bfloat16,
        )
        self.parameter_generator = ParameterGenerator(self.parameter_generator_config)
        self.parameter_generator.load_state_dict( torch.load("/home/cse_h2/dzj0055/HYWUChat-R1.5/ckpt/FINISHED/pg_weights.bin") )
        self.parameter_generator = self.parameter_generator.cuda()
        self.parameter_generator.eval()
        
        self.model_init_kwargs = {}
        self.model_init_kwargs['use_cache'] = True
        self.model_init_kwargs["torch_dtype"] = torch.bfloat16
        self.model_init_kwargs["attn_implementation"] = "flash_attention_2"
        #model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        self.model: Qwen2VLForConditionalGeneration = inject_pg(
            self.parameter_generator.config, Qwen2VLForConditionalGeneration
        ).from_pretrained(self.model_id, **self.model_init_kwargs)
        self.model.enable_pg()
        self.model = self.model.cuda()
        self.model.eval()

        self.processing_class = AutoProcessor.from_pretrained(self.model_id)
        pad_token_id = self.processing_class.tokenizer.pad_token_id
        self.processing_class.pad_token_id = pad_token_id
        self.processing_class.eos_token_id = self.processing_class.tokenizer.eos_token_id
        if "Qwen" in self.model_id or "Qwen2.5-VL" in self.model_id:
            max_pixels: Optional[int] = 12845056
            min_pixels: Optional[int] = 3136
            self.processing_class.image_processor.max_pixels = max_pixels
            self.processing_class.image_processor.min_pixels = min_pixels

        self.generation_config = GenerationConfig(
            max_new_tokens=256, # TODO: Make it a variable
            do_sample=True,  
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )


    @torch.no_grad()
    def generate(
        self,
        inputs,
        seed: int = 42,
        verbose: int = 2,
    ) -> list[str]:
        # Generate completions        
        # messages = [{"role": "user", "content": [{"type": "video", "video": inputs[0]["video_path"], "total_pixels": 3584 * 28 * 28, "min_pixels": 16 * 28 * 28,},]}]
        # image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
        if 'video_path' in inputs[0]:
            # messages = [{"role": "user", "content": [{"type": "video", "video": inputs[0]["video_path"], "total_pixels": 3584 * 28 * 28, "min_pixels": 16 * 28 * 28,},]}]
            # image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
            video_inputs = inputs[0]['video_inputs'][0]
            video_kwargs = inputs[0]['video_kwargs'][0]
            fps_inputs = video_kwargs['fps']
            image_inputs = None
        # try:
        # [{"role": "user", "content": [{"type": "image", "image": example["image_path"]}]}]
        elif 'image_path' in inputs[0]:
            # messages = [{"role": "user", "content": [{"type": "image", "image": inputs[0]["image_path"]}]}]
            # image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
            image_inputs = inputs[0]['image_inputs'][0]
            video_inputs = None
            fps_inputs = None
        # except Exception as e:
        #     video_inputs = None
        prompts = [self.make_conversation_video(example) for example in inputs]
        # print(prompts)
        prompts_text = [self.processing_class.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
        #breakpoint()

        prompt_inputs = self.processing_class(
            text=prompts_text, 
            images=image_inputs, 
            videos=video_inputs, 
            fps=fps_inputs, 
            padding=True, 
            return_tensors="pt", 
            padding_side="left", 
            add_special_tokens=False,
        )
        prompt_inputs = self._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        if 'pixel_values_videos' in prompt_inputs:
            pixel_values_videos = prompt_inputs["pixel_values_videos"]
            video_grid_thw = prompt_inputs["video_grid_thw"]
        elif 'pixel_values' in prompt_inputs:
            pixel_values = prompt_inputs["pixel_values"]
            image_grid_thw = prompt_inputs["image_grid_thw"]

        # Step 2: Extract BxD hidden states
        condition = self._extract_condition(
            model=self.model,
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            pixel_values=None,
            image_grid_thw=None,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        #breakpoint() # verified that condition extracted from model as HY WU expects

        # Step 3: PG generates LoRA parameters
        pg_state_dict = self._build_pg_state_dict(condition)
        #breakpoint() # verified that dimensions align with llm
        self.model.set_pg_state_dict(pg_state_dict)
        # Step 4: Generate with PG-LoRA
        prompt_completion_ids = self.model.generate(
            **prompt_inputs,
            generation_config=self.generation_config,
            use_cache=False,
        )
        #breakpoint()
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        #breakpoint()
        self.model.clear_pg_state_dict()
        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        #breakpoint()
        return completions
    
    def make_conversation_video(self, example):
        # if example['data_type'] == 'gqa':
        #     example_prompt = QUESTION_TEMPLATE_GQA.replace("[QUESTION]", example["problem"]["question"])
        #     example_prompt = example_prompt.replace("[OPTION]", str(example["problem"]["options"]))
        # # print(example_prompt)
        # elif example['data_type'] == 'tg':
        #     example_prompt = QUESTION_TEMPLATE_TG.replace("[EVENT]", example["problem"])
        # elif example['data_type'] == 'tracking':
        #     example_prompt = QUESTION_TEMPLATE_TRACK.replace("[OBJECT]", example["problem"]["object"])
        #     example_prompt = example_prompt.replace("[START]", str(example["problem"]["start"]))
        
        example_prompt = self.SYSTEM_PROMPT
        #example_prompt = example_prompt.replace("[QUESTION]", example["problem"]["question"])
        if 'options' in example["problem"]:
            example_prompt = example_prompt.replace("[OPTION]", str(example["problem"]["options"]))
        if 'video_path' in example:
             
            return [
                    # {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                            {
                                "type": "video", 
                                "video": example["video_path"], 
                                "total_pixels": 3584 * 28 * 28, 
                                "min_pixels": 16 * 28 * 28,
                            },
                            {"type": "text", "text": example_prompt}
                        ]
                    },
                ]
        else:
            return [
                    # {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                            {"type": "image", 
                                "image": example["image_path"], 
                            },
                            {"type": "text", "text": example_prompt},
                            
                        ]
                    },
                ]
    
    def _extract_condition(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        with torch.no_grad():
            video_embeds = model.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0)
        condition = video_embeds.unsqueeze(0).unsqueeze(1) # B,1,T,D
        #breakpoint()
        return condition
    
    def _build_pg_state_dict(self, condition):
        #breakpoint()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pg_output = self.parameter_generator(condition=condition)
        #breakpoint()
        return pg_output
    
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )

        return inputs

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": "cuda"}
            return data.to(**kwargs)
        return data