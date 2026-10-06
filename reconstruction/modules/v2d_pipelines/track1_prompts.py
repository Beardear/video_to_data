"""Typed Track 1 preparation requests and pixel-space SAM2 prompt checks."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

from v2d.common.datatypes import Sam2Prompts
from v2d.pipelines.track1_preflight import Track1Episode


@dataclass(frozen=True)
class PreparationRequest:
    episode_index: int
    prompts_path: Path
    reference_frame: int = 0
    human_id: int = 0
    object_id: int = 1


def preparation_requests(manifest: Path, episodes: dict[int, Track1Episode]) -> dict[int, PreparationRequest]:
    payload = json.loads(manifest.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != "v2d.track1.preparation.v1":
        raise ValueError("Expected a v2d.track1.preparation.v1 manifest")
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("Preparation episodes must be a list")
    requests = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"episode_index", "prompts_path", "reference_frame", "human_id", "object_id"}:
            raise ValueError("Unknown preparation request fields")
        index = row.get("episode_index")
        if type(index) is not int or index not in episodes or index in requests:
            raise ValueError("Preparation episode indexes must be known and unique")
        prompt_path = row.get("prompts_path")
        if not isinstance(prompt_path, str) or not prompt_path.strip():
            raise ValueError(f"episode {index}: prompts_path is required")
        request = PreparationRequest(index, (manifest.parent / prompt_path).resolve(),
            row.get("reference_frame", 0), row.get("human_id", 0), row.get("object_id", 1))
        if (type(request.reference_frame) is not int or not 0 <= request.reference_frame < episodes[index].expected_frames
                or any(type(i) is not int or i < 0 for i in (request.human_id, request.object_id))
                or request.human_id == request.object_id):
            raise ValueError(f"episode {index}: invalid reference frame or role IDs")
        requests[index] = request
    return requests


def validated_prompts(request: PreparationRequest, episode: Track1Episode,
                      image_size: tuple[int, int]) -> Sam2Prompts:
    """Validate geometry/timeline and resolve mask paths; semantics need visual review."""
    import numpy as np
    from PIL import Image

    payload = json.loads(request.prompts_path.read_text())
    prompts = Sam2Prompts.from_dict(payload)
    ids = {request.human_id, request.object_id}
    width, height = image_size
    if not prompts.prompts or {p.object_id for p in prompts.prompts} != ids:
        raise ValueError("SAM2 prompts must cover exactly the configured human and object IDs")
    def coordinate(value):
        return type(value) in (int, float) and math.isfinite(value)
    for raw, prompt in zip(payload["prompts"], prompts.prompts, strict=True):
        if (type(prompt.frame_index) is not int or not 0 <= prompt.frame_index < episode.expected_frames
                or type(prompt.object_id) is not int):
            raise ValueError("Prompt frame/index must be an integer within the episode")
        role = "human" if prompt.object_id == request.human_id else "object"
        if "role" in raw and raw["role"] != role:
            raise ValueError("Prompt role label disagrees with its configured object ID")
        if sum((bool(prompt.mask_path), prompt.box is not None, bool(prompt.points))) != 1:
            raise ValueError("Each prompt must provide exactly one of a box, points, or a mask")
        if prompt.box is not None:
            box = prompt.box
            if (not all(coordinate(v) for v in (box.x0, box.y0, box.x1, box.y1))
                    or not 0 <= box.x0 < box.x1 <= width or not 0 <= box.y0 < box.y1 <= height):
                raise ValueError("Prompt box must be nonempty and inside the video in pixel coordinates")
        if prompt.points:
            if (not prompt.point_labels or len(prompt.point_labels) != len(prompt.points)
                    or any(type(v) is not int or v not in (0, 1) for v in prompt.point_labels)
                    or 1 not in prompt.point_labels):
                raise ValueError("Prompt points need matching binary labels and a positive point")
            if any(not coordinate(p.x) or not coordinate(p.y) or not 0 <= p.x < width
                   or not 0 <= p.y < height for p in prompt.points):
                raise ValueError("Prompt points must be inside the video in pixel coordinates")
        if prompt.mask_path:
            path = (request.prompts_path.parent / prompt.mask_path).resolve()
            with Image.open(path) as image:
                mask = np.asarray(image)
            if mask.shape != (height, width) or not np.any(mask) or not np.isin(mask, [0, 1, 255]).all():
                raise ValueError("Prompt mask must be nonempty, binary and match the video dimensions")
            prompt.mask_path = str(path)
    return prompts
