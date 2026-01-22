from pathlib import Path

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class OntologyConfig(BaseModel):
    """Defines world model for teacher and student models in Object Detection FT process.
    Serves as ontology definition for predicates in VLM finetune too (optional param).
    """

    objects: dict[str, str] | None = Field(
        None, description="Map of more specific to more general"
    )
    predicates: list[str] | None = Field(
        None, description="List of predicates in VLM SGG finetuning."
    )

    @model_validator(mode="after")
    def check_at_least_one_ontology(self):
        if self.objects and self.predicates:
            return self
        raise ValueError(
            "At least one ontology is required (dict[str, str]). Specify objects or predicates (list[str])."
        )


class DataPathConfig(BaseModel):
    input_dir: Path = Field(None, description="Path to input data directory.")
    output_dir: Path = Field(None, description="Path to output data directory.")

    @model_validator(mode="after")
    def check_and_sanitize_directories(self):
        if not self.input_dir.exists():
            try:
                self.input_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(
                    f"Input directory path does not exist and cannot be created: {self.input_dir}"
                ) from e

        if not self.input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {self.input_dir}")

        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValueError(
                    f"Output directory path does not exist and cannot be created: {self.output_dir}"
                ) from e

        if not self.output_dir.is_dir():
            raise ValueError(f"Output path is not a directory: {self.output_dir}")

        return self
