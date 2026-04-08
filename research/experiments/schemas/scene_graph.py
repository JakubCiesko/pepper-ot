from pydantic import BaseModel
from pydantic import Field


class ImagePredicatesAttributes(BaseModel):
    predicates: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)


class GeneralPredicates(BaseModel):
    predicates: list[str] = Field(default_factory=list)


class GeneralAttributes(BaseModel):
    attributes: list[str] = Field(default_factory=list)


class SceneGraphRelation(BaseModel):
    sub: str
    rel: str
    obj: str


class SceneGraphDraft(BaseModel):
    relationships: list[SceneGraphRelation] = Field(default_factory=list)
