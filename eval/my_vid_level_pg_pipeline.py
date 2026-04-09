from typing import Optional, Union, Any, Dict
from collections.abc import Iterator, Mapping

import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig, PreTrainedTokenizerBase

from open_r1.vlm_modules import InvernVLModule

from wu.config import ParameterGeneratorConfig
from wu.generator import ParameterGenerator
from wu.mixin import inject_pg

class PGPipeline:
    def __init__(
        self,
        device_map: str = "auto",
    ):
        self.model_id = "OpenGVLab/InternVL2-8B"
        self.vlm_module = InvernVLModule()
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
            input_dim=4096,
            dim_accumulation=4,
            d_model=512,
            head_dim=512,
            num_base_model_layers=32,
            output_dim=4096,
            num_pg_layers=16,
            prefix="language_model.model.layers.",
            pg_mapping={
                # wqkv is fused QKV: in=4096, out=6144 (q=4096, k=1024, v=1024)
                "attention.wqkv": {"lora_A_dim": 4096, "lora_B_dim": 6144},
                "attention.wo":   {"lora_A_dim": 4096, "lora_B_dim": 4096},
                # MLP: w1 and w3 are gate/up, w2 is down
                "feed_forward.w1": {"lora_A_dim": 4096,  "lora_B_dim": 14336},
                "feed_forward.w3": {"lora_A_dim": 4096,  "lora_B_dim": 14336},
                "feed_forward.w2": {"lora_A_dim": 14336, "lora_B_dim": 4096},
            },
            torch_dtype=torch.bfloat16,
        )
        self.parameter_generator = ParameterGenerator(self.parameter_generator_config)
        self.parameter_generator.load_state_dict( torch.load("/aiau001_scratch/dzj0055/HYWUVLM-R1/my_output/InternVL2-8B-GRPO-VAD-lr5e-7/pg_weights.bin") )
        self.parameter_generator = self.parameter_generator.cuda()
        self.parameter_generator.eval()
        
        self.model_init_kwargs = {
            'torch_dtype': 'bfloat16', 
            'trust_remote_code': True,
            'use_flash_attn': True,
            'use_cache': True
        }
        #model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        model_cls = self.vlm_module.get_model_class(self.model_id, self.model_init_kwargs)
        internvl_model_class = model_cls.from_pretrained(self.model_id, trust_remote_code=True).__class__
        self.model = inject_pg(
            self.parameter_generator.config, internvl_model_class
        ).from_pretrained(self.model_id, **self.model_init_kwargs)
        self.model.enable_pg()
        self.model = self.model.cuda()
        self.model.eval()
        #breakpoint()

        
        processing_cls = self.vlm_module.get_processing_class()
        processing_class = processing_cls.from_pretrained(self.model_id, trust_remote_code=self.model_init_kwargs.get("trust_remote_code", None))
        kwargs = {'max_pixels': 12845056, 'min_pixels': 3136, 'max_anyres_num': 1}
        for component, processing_keyword in self.vlm_module.get_custom_processing_keywords():
            if processing_keyword in kwargs:
                # If we cannot find component in processing_class, return the processing_class itself
                processing_component = getattr(processing_class, component, processing_class)
                setattr(processing_component, processing_keyword, kwargs[processing_keyword])
        if getattr(processing_class, "tokenizer",  None) is not None:
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        else:
            assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
            pad_token_id = processing_class.pad_token_id
        #breakpoint()

        self.vlm_module.post_model_init(self.model, processing_class)
        self.processing_class = processing_class
        #breakpoint()

        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=True,  
            temperature=1,
            pad_token_id=pad_token_id,
            eos_token_id=self.vlm_module.get_eos_token_id(processing_class)
        )

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    @torch.no_grad()
    def generate(
        self,
        inputs,
        seed: int = 42,
        verbose: int = 2,
    ) -> list[str]:
        # Generate completions        
        device = "cuda"
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
        #breakpoint()
        # TODO: Change to video path
        # Handle both pre-loaded images and image paths
        images = []
        for x in inputs:
            if "image" in x:
                imgs = self._get_key_from_inputs(x, "image")
            elif "image_path" in x and x["image_path"] is not None:
                imgs = [PIL.Image.open(p) for p in self._get_key_from_inputs(x, "image_path")]
            elif "video" in x and x["video"] is not None:
                imgs = [i for i in self._get_key_from_inputs(x, "video")]
            else:
                imgs = []
            #breakpoint()
            for img in imgs:
                try:
                    # Ensure minimum dimensions of 28 pixels
                    w, h = img.size
                    if w < 28 or h < 28:
                    # Calculate new dimensions maintaining aspect ratio
                        if w < h:
                            new_w = 28
                            new_h = int(h * (28/w))
                        else:
                            new_h = 28
                            new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                except:
                    pass
                images.append(img)
                
        #breakpoint()
        prompt_inputs, additional_output = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        #breakpoint()
        prompt_inputs = self._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # image_grid_thw may be needed for the reward function

        if additional_output is not None:
            assert len(additional_output) == len(inputs)
            for i, (input_i, additional_output_i) in enumerate(zip(inputs, additional_output)):
                input_i.update(additional_output_i)

        prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(torch.bfloat16)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generate_returned_result = self.model.generate(
                **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()}, 
                generation_config=self.generation_config
            )
        #breakpoint()

        prompt_length = prompt_ids.size(1)
        if not self.vlm_module.is_embeds_input():
            prompt_completion_ids = generate_returned_result
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
        else:
            # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
            # So the returned result of the `generate` method only contains the completion ids
            completion_ids = generate_returned_result
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        #breakpoint()

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        #breakpoint()
        return completions
    
    def inject_parameters(self, inputs):
        device = "cuda"
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
        #breakpoint()
        # TODO: Change to video path
        # Handle both pre-loaded images and image paths
        images = []
        for x in inputs:
            if "image" in x:
                imgs = self._get_key_from_inputs(x, "image")
            elif "image_path" in x and x["image_path"] is not None:
                imgs = [PIL.Image.open(p) for p in self._get_key_from_inputs(x, "image_path")]
            elif "video" in x and x["video"] is not None:
                imgs = [i for i in self._get_key_from_inputs(x, "video")]
            else:
                imgs = []
            #breakpoint()
            for img in imgs:
                try:
                    # Ensure minimum dimensions of 28 pixels
                    w, h = img.size
                    if w < 28 or h < 28:
                    # Calculate new dimensions maintaining aspect ratio
                        if w < h:
                            new_w = 28
                            new_h = int(h * (28/w))
                        else:
                            new_h = 28
                            new_w = int(w * (28/h))
                    img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                except:
                    pass
                images.append(img)
                
        #breakpoint()
        prompt_inputs, additional_output = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = self._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        #breakpoint()
        # image_grid_thw may be needed for the reward function

        if additional_output is not None:
            assert len(additional_output) == len(inputs)
            for i, (input_i, additional_output_i) in enumerate(zip(inputs, additional_output)):
                input_i.update(additional_output_i)

        #breakpoint()
        prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].to(torch.bfloat16)
        
        # Generate completions
        # NOTE: This will only work when there is one video being analyzed at a time
        condition = self._extract_condition(
            model=self.model,
            pixel_values_videos=prompt_inputs['pixel_values'],
        )
        #breakpoint()
        # Step 3: PG generates LoRA parameters
        pg_state_dict = self._build_pg_state_dict(condition)
        #breakpoint() # verified that dimensions align with llm
        self.model.set_pg_state_dict(pg_state_dict)
        #breakpoint()
        print("INJECTED PARAMETERS")
    
    def clear_injected_parameters(self):
        self.model.clear_pg_state_dict()
        #breakpoint()
        print("CLEARED PARAMETERS")
   
    def _extract_condition(
        self, model,
        pixel_values_videos=None,
    ):
        with torch.no_grad():
            #breakpoint()
            video_embeds = model.extract_feature(pixel_values_videos)
            #breakpoint()
            video_embeds = torch.flatten(video_embeds, start_dim=0, end_dim=1)
        #breakpoint()
        condition = video_embeds.unsqueeze(0).unsqueeze(1) # B,1,T,D
        #breakpoint()
        return condition

    def _build_pg_state_dict(self, condition):
        #breakpoint()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pg_output = self.parameter_generator(condition=condition)
        #breakpoint()
        return pg_output

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
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