import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from PIL import Image
import torch
import torchvision.transforms as T

VG_CLASSES = [
    "N/A",
    "airplane",
    "animal",
    "arm",
    "bag",
    "banana",
    "basket",
    "beach",
    "bear",
    "bed",
    "bench",
    "bike",
    "bird",
    "board",
    "boat",
    "book",
    "boot",
    "bottle",
    "bowl",
    "box",
    "boy",
    "branch",
    "building",
    "bus",
    "cabinet",
    "cap",
    "car",
    "cat",
    "chair",
    "child",
    "clock",
    "coat",
    "counter",
    "cow",
    "cup",
    "curtain",
    "desk",
    "dog",
    "door",
    "drawer",
    "ear",
    "elephant",
    "engine",
    "eye",
    "face",
    "fence",
    "finger",
    "flag",
    "flower",
    "food",
    "fork",
    "fruit",
    "giraffe",
    "girl",
    "glass",
    "glove",
    "guy",
    "hair",
    "hand",
    "handle",
    "hat",
    "head",
    "helmet",
    "hill",
    "horse",
    "house",
    "jacket",
    "jean",
    "kid",
    "kite",
    "lady",
    "lamp",
    "laptop",
    "leaf",
    "leg",
    "letter",
    "light",
    "logo",
    "man",
    "men",
    "motorcycle",
    "mountain",
    "mouth",
    "neck",
    "nose",
    "number",
    "orange",
    "pant",
    "paper",
    "paw",
    "people",
    "person",
    "phone",
    "pillow",
    "pizza",
    "plane",
    "plant",
    "plate",
    "player",
    "pole",
    "post",
    "pot",
    "racket",
    "railing",
    "rock",
    "roof",
    "room",
    "screen",
    "seat",
    "sheep",
    "shelf",
    "shirt",
    "shoe",
    "short",
    "sidewalk",
    "sign",
    "sink",
    "skateboard",
    "ski",
    "skier",
    "sneaker",
    "snow",
    "sock",
    "stand",
    "street",
    "surfboard",
    "table",
    "tail",
    "tie",
    "tile",
    "tire",
    "toilet",
    "towel",
    "tower",
    "track",
    "train",
    "tree",
    "truck",
    "trunk",
    "umbrella",
    "vase",
    "vegetable",
    "vehicle",
    "wave",
    "wheel",
    "window",
    "windshield",
    "wing",
    "wire",
    "woman",
    "zebra",
]

VG_REL_CLASSES = [
    "__background__",
    "above",
    "across",
    "against",
    "along",
    "and",
    "at",
    "attached to",
    "behind",
    "belonging to",
    "between",
    "carrying",
    "covered in",
    "covering",
    "eating",
    "flying in",
    "for",
    "from",
    "growing on",
    "hanging from",
    "has",
    "holding",
    "in",
    "in front of",
    "laying on",
    "looking at",
    "lying on",
    "made of",
    "mounted on",
    "near",
    "of",
    "on",
    "on back of",
    "over",
    "painted on",
    "parked on",
    "part of",
    "playing",
    "riding",
    "says",
    "sitting on",
    "standing on",
    "to",
    "under",
    "using",
    "walking in",
    "walking on",
    "watching",
    "wearing",
    "wears",
    "with",
]


@dataclass
class RelTRImagePrediction:
    objects: list[dict[str, Any]]
    relationships: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "objects": self.objects,
            "relationships": self.relationships,
        }


def _build_args(dataset: str, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        lr_backbone=1e-5,
        dataset=dataset,
        backbone="resnet50",
        dilation=False,
        position_embedding="sine",
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.1,
        nheads=8,
        num_entities=100,
        num_triplets=200,
        pre_norm=False,
        aux_loss=True,
        device=device,
        set_cost_class=1.0,
        set_cost_bbox=5.0,
        set_cost_giou=2.0,
        set_iou_threshold=0.7,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        rel_loss_coef=1.0,
        eos_coef=0.1,
        return_interm_layers=False,
    )


def _rescale_boxes(boxes_cxcywh: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    img_w, img_h = size
    x_c, y_c, w, h = boxes_cxcywh.unbind(1)
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    xyxy = xyxy * torch.tensor(
        [img_w, img_h, img_w, img_h], dtype=torch.float32, device=xyxy.device
    )
    return xyxy


def predict_image(
    repo_root: Path,
    checkpoint_path: Path,
    image_path: Path,
    dataset: str = "vg",
    device: str = "cuda",
    threshold: float = 0.3,
    topk: int = 100,
) -> RelTRImagePrediction:
    sys.path.insert(0, str(repo_root))
    from models import build_model  # pylint: disable=import-error

    if dataset != "vg":
        raise ValueError("Current adapter supports dataset='vg' label space only.")

    transform = T.Compose(
        [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    args = _build_args(dataset=dataset, device=device)
    model, _, _ = build_model(args)

    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)

    image = Image.open(str(image_path)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    rel_probs = outputs["rel_logits"].softmax(-1)[0, :, :-1]
    sub_probs = outputs["sub_logits"].softmax(-1)[0, :, :-1]
    obj_probs = outputs["obj_logits"].softmax(-1)[0, :, :-1]

    keep = torch.logical_and(
        rel_probs.max(-1).values > threshold,
        torch.logical_and(
            sub_probs.max(-1).values > threshold,
            obj_probs.max(-1).values > threshold,
        ),
    )

    keep_idx = torch.nonzero(keep, as_tuple=True)[0]
    if keep_idx.numel() == 0:
        return RelTRImagePrediction(objects=[], relationships=[])

    score = (
        rel_probs[keep_idx].max(-1).values
        * sub_probs[keep_idx].max(-1).values
        * obj_probs[keep_idx].max(-1).values
    )
    sorted_idx = keep_idx[torch.argsort(-score)[:topk]]

    sub_boxes = _rescale_boxes(outputs["sub_boxes"][0, sorted_idx], image.size).cpu()
    obj_boxes = _rescale_boxes(outputs["obj_boxes"][0, sorted_idx], image.size).cpu()

    objects: list[dict[str, Any]] = []
    obj_index: dict[tuple[str, tuple[int, int, int, int]], int] = {}

    def register_object(label_idx: int, box: torch.Tensor, score_value: float) -> int:
        label_idx_i = int(label_idx)
        label = (
            VG_CLASSES[label_idx_i]
            if 0 <= label_idx_i < len(VG_CLASSES)
            else str(label_idx_i)
        )
        b = tuple(int(round(v)) for v in box.tolist())
        key = (label, b)
        if key in obj_index:
            return obj_index[key]
        oid = len(objects)
        objects.append(
            {
                "id": oid,
                "label": label,
                "bbox": [float(v) for v in box.tolist()],
                "score": float(score_value),
            }
        )
        obj_index[key] = oid
        return oid

    relationships: list[dict[str, Any]] = []
    for i, q_idx in enumerate(sorted_idx.tolist()):
        rel_id = int(rel_probs[q_idx].argmax().item())
        sub_id = int(sub_probs[q_idx].argmax().item())
        obj_id = int(obj_probs[q_idx].argmax().item())

        if rel_id == 0:
            continue

        rel_label = (
            VG_REL_CLASSES[rel_id] if 0 <= rel_id < len(VG_REL_CLASSES) else str(rel_id)
        )
        rel_label = rel_label.replace(" ", "_")

        sub_obj_id = register_object(
            sub_id, sub_boxes[i], float(sub_probs[q_idx].max().item())
        )
        obj_obj_id = register_object(
            obj_id, obj_boxes[i], float(obj_probs[q_idx].max().item())
        )

        relationships.append(
            {
                "sub": f"{objects[sub_obj_id]['label']}_{sub_obj_id}",
                "rel": rel_label,
                "obj": f"{objects[obj_obj_id]['label']}_{obj_obj_id}",
                "score": float(rel_probs[q_idx].max().item()),
            }
        )

    return RelTRImagePrediction(objects=objects, relationships=relationships)
