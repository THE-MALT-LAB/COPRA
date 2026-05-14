# HYWUVLM-R1

## Installation

- Install Python 3.10
```shell
conda create -p $PATH_TO_ENV
conda activate $PATH_TO_ENV
conda install python=3.10
```
- Install cuda 12.4
```shell
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
```
- [OPTIONAL] If you get iJIT_NotifyEvent error when loading torch,
```shell
conda install -y "mkl<2024.1" "intel-openmp<2024.1"
```
- Install torch 2.5.1
```shell
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```
- [OPTIONAL] install flash attention
```shell
pip install ninja psutil==7.2.2
pip install --no-build-isolation flash_attn==2.7.4.post1
```

- Install dependencies for VLM-R1
```shell
cd VLM-R1/
bash setup.sh
pip install transformers==4.49.0
pip install decord
pip install timm
```
NOTE: VLM-R1 will NOT work with higher transformers version v4.57.1 (from HY-WU/requirements.py) without more changes! InternLM2 exposes generate without inheriting from GenerationMixIn

## Evaluation

Please download **UCF-Crime** and **XD-Violence** from the official author mirrors. After downloading, create symbolic links to the dataset directories inside this repository so the evaluation scripts can find them consistently.

### UCF-Crime

To evaluate on UCF-Crime, run:

```bash
sh eval_vid_level.sh
```

### XD-Violence

To evaluate on XD-Violence, run:

```bash
sh eval_vid_level_xd.sh
```

## TODO

- [x] Update VLM-R1 to utilize UCF-Crime frames and annotations
- [x] Update VLM-R1 to utilize UCF-Crime video-level labels
- [x] Test GRPO on VLM-R1
- [x] Test HY-WU parameter generation
- [x] Integrate parameter generation into GRPO
    - generate -> inject -> respond -> reinforceE
- [x] Add parameter generator to InternVL2 backbone
    - Ensure parameter generator incorporates vision (video frames) alone
    - check wu/pipeline.py:generate
- [x] Add optimizer for parameter generator
- [x] Check if gradients pass to parameter generator NOT backbone
    - log parameter generator gradient l2 norm with wandb over time?
- [x] Test response generation with parameter generation
- [x] Test response generation with parameter generation after 2 GRPO updates
- [x] Check if generated parameters are non-trivial over training
    - log generated parameter weight l2 norm with wandb over time
- [x] added saving for parameter generator weights
- [x] Check if GRPO with parameter generation leads to stable rewards/losses over training   
- [x] Add evaluation script from VERAS
