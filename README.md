# HYWUChat-R1.5

## Installation

- Install Python 3.11
```shell
conda create -p $PATH_TO_ENV
conda activate $PATH_TO_ENV
conda install python=3.11
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
pip install ninja
pip install psutil==6.0.0
pip install --no-build-isolation flash_attn==2.5.9.post1
```

- Install dependencies for VideoChat-R1.5
```shell
cd VideoChat-R1/
pip install -r requirements.txt
```

- Install dependencies for HY-WU
```shell
cd HY-WU/
pip install -r requirements.txt
```

## TODO

- [x] Update VideoChat-R1.5 to utilize UCF-Crime frames and annotations
- [x] Update VideoChat-R1.5 to utilize UCF-Crime video-level labels
- [x] Test GRPO on VideoChat-R1.5
- [x] Test HY-WU parameter generation
- [x] Integrate parameter generation into GRPO
    - generate -> inject -> respond -> reinforce
- [] Add parameter generator to Qwen2-VL backbone
    - Ensure parameter generator incorporates vision (video frames) and textual information (prompt)
    - check wu/pipeline.py:generate
- [] Check if gradients pass to parameter generator NOT backbone
    - log parameter generator gradient l2 norm with wandb over time?
- [] Test response generation with parameter generation
- [] Test response generation with parameter generation after 2 GRPO updates
- [] Check if generated parameters are non-trivial over training
    - log generated parameter weight l2 norm with wandb over time
- [] Check if GRPO with parameter generation leads to stable rewards/losses over training   