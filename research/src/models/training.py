from pathlib import Path

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from unsloth import is_bfloat16_supported


class LoRAConfig(BaseModel):
    """LoRA Config for VLM training"""

    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


class UnslothTrainingConfig(BaseModel):
    """Hyperparameters for the SFTTrainer."""

    base_model_id: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
    output_dir: str = "checkpoints"

    # Training Loop
    batch_size: int = Field(2, gt=0)
    gradient_accumulation_steps: int = Field(4, gt=0)
    max_steps: int = 200
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    fp16: bool | None = None
    bf16: bool | None = None
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    seed: int = 42

    # Export
    export_quantization: str = "q4_k_m"
    hf_hub_id: str | None = "qwen3-vl-8b-som-sgg"

    @model_validator(mode="after")
    def resolve_precision(self):
        if self.fp16 is None and self.bf16 is None:
            is_bf16 = is_bfloat16_supported()
            self.bf16 = is_bf16
            self.fp16 = not is_bf16
        return self


class VLMTrainerConfig(BaseModel):
    """The Root Config Object for the VLM script."""

    lora: LoRAConfig
    training: UnslothTrainingConfig
    dataset_path: Path = Field(
        ..., description="Path to jsonl dataset with generated scenegraphs"
    )


class DetectorTrainerConfig(BaseModel):
    """Config for fine-tuning RT-DETR"""

    base_model: str = "rtdetr-l.pt"
    project_name: str = "pepper_thesis_detector"

    # Data
    dataset_yaml: Path = Field(
        ..., description="Path to the data.yaml created by distillation"
    )

    # Hyperparams
    epochs: int = Field(50, gt=0)
    imgsz: int = 640
    batch_size: int = 8
    device: str = "0"  # GPU ID

    # Optimizer
    optimizer: str = "auto"
    lr0: float = 0.0001

    @model_validator(mode="after")
    def validate_yaml(self):
        if not self.dataset_yaml.exists():
            raise ValueError(f"Dataset YAML not found at {self.dataset_yaml}")
        return self
