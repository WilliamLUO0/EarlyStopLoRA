# Towards Personalized AI: Early-stopping Low-Rank Adaptation of Foundation Models

This repository contains code implementations designed to perform early stop during LoRA fine-tuning based on CS-Fluctuation. The implementation is built upon [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts.git) and [artidoro/qlora](https://github.com/artidoro/qlora.git).

## Overview
Foundation models, such as Latent Diffusion Models and Generative Pre-trained Transformers, trained on broad data have shown impressive results in various downstream applications. Fine-tuning a pre-trained foundation model is an affordable way to customize it on small and personalized data. However, the non-AI experts often struggle with the hyperparameter configurations and sometimes encounter the overfitting issue without even realizing it. To mitigate this issue, we introduce a new monitoring metric (CS-Fluctuation) to facilitate early stopping the fine-tuning process. Specifically, we leverage Low-Rank Adaptation (LoRA) to fit the small scale of the personalized data while monitoring the cosine similarity of the parameter changes between the LoRA branch and its corresponding layer. When the changes become steady, we observe the onset of overfitting issue which becomes increasingly severe as fine-tuning progresses. Empirically, we leverage various types of personalized data to conduct customization experiments on both vision and language foundation models, which corroborates the effectiveness of CS-Fluctuation in early stopping the LoRA fine-tuning. 

## Installation
For environment setup and configurations:

- If you wish to fine-tune Latent Diffusion Models (LDMs), please refer to the README of [sd-scripts](https://github.com/kohya-ss/sd-scripts.git) for detailed environment configurations and setups.

- For fine-tuning Large Language Models (LLMs), please follow the instructions provided in the README of [qlora](https://github.com/artidoro/qlora.git).

It is strongly recommended to create two separate virtual environments.

## Getting started

### Early Stop LoRA Fine-tuning for LDMs

The `es_lora_ldm.py` script located under `lora-ldm` is modified based on `train_network.py` from `sd-scripts`, incorporating the early stop strategy base on CS-Fluctuation.

#### 1. Choose a Base Model
Firstly, select a base model. Models can be downloaded from the following websites:
    - [CivitAI](https://civitai.com/)
    - [Hugging Face](https://huggingface.co/)

#### 2. Dataset Preprocessing
Four datasets are provided under the `dataset` folder, including real portraits, celebrity stills, landscapes of Queenstown in Auckland, and architectures of the Forbidden City. Then you need to resize or crop the images to the set resolution, e.g. 512*512. After that, we recommend to use Tagger to generate tags for images, such as [SmilingWolf/wd-v1-4-moat-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2), then further manual adjustments are also necessary.

#### 3. Fine-tune LDMs
Now, place a copy of `es_lora_ldm.py` into `lora-ldm`, you can then train the LoRA model using `early_stop_lora_ldm.sh`! Note that the window size `$M$` is usually set to the length of one epoch, calculated as `number of images * repeat / cs_interval`, where `cs_interval` denotes the number of steps between each computation of cosine similarity and CS-Fluctuation. If you wish to observe the full training process without early stopping, simply omit the --early_stop option.

### Early Stop LoRA Fine-tuning for LLMs

The `es_lora_llm.py` script in `lora-llm` is a modified version of `qlora.py` in the `qlora`, with the early stop strategy introduced.

#### 1. Dataset Preparation
Firstly, download `data.tar` and `mmlu.py` from the [MMLU dataset](https://huggingface.co/datasets/cais/mmlu/tree/main) and place them in the `qlora/data/mmlu` folder.

#### 2. Fine-tune LLMs
Now, you can directly use `easy_stop_lora_llm` to fine-tune LLMs! If you wish to observe the full training process without early stopping, simply omit the --early_stop option.



