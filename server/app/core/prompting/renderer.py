from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptRenderContext:
    context: str | None = None
    caption: str | None = None
    captions_recent: str | None = None
    predicates: Iterable[Any] | str | None = None
    objects: Any | None = None
    extra: dict[str, Any] | None = None

    def to_template_values(self) -> dict[str, Any]:
        if isinstance(self.predicates, str):
            predicates_text = self.predicates
        else:
            predicates_text = ", ".join(str(item) for item in (self.predicates or []))
        if isinstance(self.objects, str):
            objects_text = self.objects
        else:
            objects_text = str(self.objects or "")
        recent = self.captions_recent or ""
        values: dict[str, Any] = {
            "context": self.context or "",
            "caption": self.caption or "",
            "captions_recent": recent,
            "caption_recent": recent,
            "predicates": predicates_text,
            "objects": objects_text,
        }
        if self.extra:
            values.update(self.extra)
        return values


def render_prompt_template(
    template: str | None,
    values: dict[str, Any] | PromptRenderContext | None = None,
) -> str:
    if not template:
        return ""
    if isinstance(values, PromptRenderContext):
        payload = values.to_template_values()
    else:
        payload = values or {}

    rendered = template
    for key, value in payload.items():
        rendered = rendered.replace("{" + key + "}", str(value or ""))
    return rendered
