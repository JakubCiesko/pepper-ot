from pydantic import BaseModel
from pydantic import Field


class ImagePredicatesAttributes(BaseModel):
    """Structured LLM output for vocabulary candidates from one image."""

    predicates: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)


class GeneralPredicates(BaseModel):
    """Structured LLM output for the consolidated predicate vocabulary."""

    predicates: list[str] = Field(default_factory=list)


class GeneralAttributes(BaseModel):
    """Structured LLM output for the consolidated attribute vocabulary."""

    attributes: list[str] = Field(default_factory=list)


class SceneGraphRelation(BaseModel):
    """One scene graph relation or unary attribute row.

    Attributes:
        sub: Subject object ID as referenced in the prompt.
        rel: Predicate or attribute label.
        obj: Object ID. Unary attributes use the same value as sub.
    """

    sub: str
    rel: str
    obj: str


class SceneGraphDraft(BaseModel):
    """Structured VLM output for draft scene graph generation."""

    relationships: list[SceneGraphRelation] = Field(default_factory=list)
