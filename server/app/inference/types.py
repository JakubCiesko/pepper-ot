from pydantic import BaseModel

# i will add scene graph types here, detection thing types, tracked obj?


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def get_rel_angle(self, image_width: int) -> float:
        """Returns relative horizontal angle (-0.5 to 0.5)."""
        return (self.centroid[0] / image_width) - 0.5
