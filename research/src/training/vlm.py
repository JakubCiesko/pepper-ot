import json
import logging
from pathlib import Path
import sys

from datasets import Dataset
import torch
from trl import SFTConfig
from trl import SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
import yaml

from research.src.models.training import VLMTrainerConfig

logger = logging.getLogger(__name__)


def load_and_format_dataset(jsonl_path: Path, system_prompt: str):
    logger.info(f"Loading dataset from {jsonl_path}...")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")
    data_entries = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)

                # Input: The SoM Image path
                image_path = entry["image"]

                # Output: The Scene Graph JSON (as a string)
                target_response = json.dumps(entry["scene_graph"], indent=2)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": system_prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": target_response}],
                    },
                ]
                data_entries.append({"messages": conversation})
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line in {jsonl_path}")

    logger.info(f"Formatted {len(data_entries)} training examples.")
    return Dataset.from_list(data_entries)


def run_vlm_training(config_path: Path):
    logger.info(f"Loading VLM config from {config_path}")
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    config = VLMTrainerConfig(**raw_config)
    logger.info(f"VLM Training Config loaded: {config.model_dump()}")

    logger.info(f"Loading Base Model: {config.training.base_model_id}")
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            config.training.base_model_id,
            load_in_4bit=config.training.load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )
    except Exception as e:
        logger.error(f"Failed to load Unsloth model. Is Unsloth installed? Error: {e}")
        sys.exit(1)

    # LoRA
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=config.lora.finetune_vision_layers,
        finetune_language_layers=config.lora.finetune_language_layers,
        finetune_attention_modules=config.lora.finetune_attention_modules,
        finetune_mlp_modules=config.lora.finetune_mlp_modules,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        random_state=config.training.seed,
        use_rslora=config.lora.use_rslora,
        loftq_config=None,
    )
    system_prompt = config.system_prompt
    logger.info(f"Loaded system prompt: {system_prompt}")
    dataset = load_and_format_dataset(config.dataset_path, system_prompt)
    logger.info("Initializing SFTTrainer...")
    gpu_stats = torch.cuda.get_device_properties(0)
    logger.info(
        f"GPU: {gpu_stats.name}. Max Memory: {gpu_stats.total_memory / 1024 ** 3:.2f} GB"
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        dataset_text_field="",  # Handled by 'messages' format
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=config.training.batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            warmup_steps=config.training.warmup_steps,
            max_steps=config.training.max_steps,
            learning_rate=config.training.learning_rate,
            fp16=config.training.fp16,
            bf16=config.training.bf16,
            logging_steps=config.training.logging_steps,
            optim=config.training.optim,
            weight_decay=config.training.weight_decay,
            lr_scheduler_type=config.training.lr_scheduler_type,
            seed=config.training.seed,
            output_dir=config.training.output_dir,
            report_to=config.training.report_to,
            # Checkpointing
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
        ),
    )

    logger.info("Starting Training...")
    trainer.train()

    logger.info(f"Saving LoRA adapters to {config.training.output_dir}")
    model.save_pretrained(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)

    logger.info("VLM Fine-tuning Complete!")
