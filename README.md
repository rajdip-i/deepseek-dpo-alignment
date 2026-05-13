# DeepSeek DPO Alignment

A experimental project for fine-tuning `deepseek-ai/deepseek-llm-7b-base` with Direct Preference Optimization (DPO) on the `Anthropic/hh-rlhf` dataset using TRL, QLoRA, LoRA, PEFT, and bitsandbytes.

## Project Overview

This project explores preference-based alignment for DeepSeek 7B using Direct Preference Optimization (DPO). Instead of modifying behavior only at inference time, this workflow fine-tunes the model on human preference pairs so it learns to prefer stronger responses over weaker ones.

The training setup uses:

- `deepseek-ai/deepseek-llm-7b-base`
- `Anthropic/hh-rlhf`
- Hugging Face `transformers`
- Hugging Face `trl`
- `peft`
- `bitsandbytes`
- QLoRA-based 4-bit training on Kaggle GPU

## Goal

The goal of this project is to align a base language model using preference data, where each example contains:

- a shared prompt
- a preferred response (`chosen`)
- a rejected response (`rejected`)

Using DPO, the model is trained to increase preference for the chosen answer relative to the rejected one, while staying close to a frozen reference model.

## Why DPO

Direct Preference Optimization is a practical alternative to a full RLHF pipeline.

Benefits:

- simpler than reward-model-plus-PPO training
- directly uses preference pairs
- stable and efficient with TRL
- works well with parameter-efficient fine-tuning such as LoRA and QLoRA

## Dataset

This project uses:

- `Anthropic/hh-rlhf`

A subset such as:

- `train[:2000]`

is used for a lightweight experimental run on Kaggle.

Each row is converted into:

```python
{
    "prompt": ...,
    "chosen": ...,
    "rejected": ...
}
```

The prompt is extracted from the shared conversation context, and the final assistant completions are separated into preferred and rejected responses.

## Training Setup

### Base Model

- `deepseek-ai/deepseek-llm-7b-base`

### Quantization

- 4-bit QLoRA
- `nf4`
- double quantization enabled

### LoRA Configuration

- `r = 8`
- `lora_alpha = 16`
- `target_modules = ["q_proj", "v_proj"]`
- `lora_dropout = 0.05`

### DPO Training Config

- `per_device_train_batch_size = 2`
- `gradient_accumulation_steps = 4`
- `num_train_epochs = 1`
- `learning_rate = 5e-5`
- `fp16 = True` or `bf16` when supported
- `logging_steps = 10`
- `save_steps = 500`

## Project Workflow

1. Load the DeepSeek 7B base model in 4-bit quantized mode.
2. Attach LoRA adapters for parameter-efficient training.
3. Load and format the HH-RLHF dataset into prompt/chosen/rejected triplets.
4. Initialize a frozen reference model.
5. Configure TRL's `DPOTrainer`.
6. Train the policy model with DPO loss.
7. Save the fine-tuned adapter.
8. Compare generations before and after training.

## Notebook

Main notebook:

- `deepseek_7b_dpo_qlora_kaggle.ipynb`

This notebook is designed to run on Kaggle GPU and includes:

- environment setup
- Hugging Face login
- tokenizer handling
- dataset formatting
- QLoRA model loading
- LoRA setup
- DPO training
- before/after generation comparison
- save/reload workflow

## Key Learnings

This project also reflects practical lessons from experimentation:

- library version mismatches in Kaggle can affect `DPOConfig` and `DPOTrainer`
- dataset formatting is critical for DPO
- tokenizer padding and EOS setup matter for stable training
- QLoRA makes 7B preference tuning feasible on limited GPU resources
- a frozen reference model is essential for the DPO objective

## Challenges Addressed

During implementation, several issues had to be handled carefully:

- TRL API differences across versions
- argument compatibility for `DPOConfig` and `DPOTrainer`
- splitting HH conversations into prompt vs response correctly
- saving adapters cleanly after training
- making the notebook robust for Kaggle runtime constraints


## How To Run

1. Open the notebook in Kaggle.
2. Enable `Internet`.
3. Enable `GPU`.
4. Add a Kaggle secret named `HF_TOKEN`.
5. Run the notebook step by step.
6. Inspect the before/after generation outputs.
7. Save and export the trained adapter.

## Repository Structure

- `DPO Trainer.ipynb` — main Kaggle DPO fine-tuning notebook
- `README.md` — project overview and instructions

## Summary

This project is a practical implementation of preference alignment for DeepSeek 7B using DPO and QLoRA. It provides a reproducible  workflow for training on Anthropic HH-RLHF and serves as a strong starting point for RL alignment and post-training research on open LLMs.
