"""Send a snapshot to an image model and keep only the repaired hole.

The viewer's texture step assumes the repaired image is pixel-aligned with the
snapshot it came from.  A general image-editing model does not honour that: it
regenerates the whole frame, so the object's pattern, lighting, silhouette and
even the output resolution can all drift.  Feeding that back wholesale would
texture the patch with a subtly different object.

By default the model's output is aligned back to the snapshot's resolution and
then composited through the hole mask that detection already produced --
outside the hole the original snapshot survives byte for byte, and only the
region that genuinely had nothing in it is taken from the model.
:func:`composite_repair` is what makes an unconstrained generator safe to use
here.  Callers that need to align or harmonize at the model's native resolution
may explicitly request the raw output as a second result; the safe composited
result remains the default contract.

The interactive reconstruction Viewer uses Volcengine Ark Seedream.  Legacy
Qwen request helpers remain in this module only for archived/offline pipeline
tools; they are not exposed by the Viewer.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import io
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urljoin, urlsplit

import numpy as np

from click_gaussian.ark_credentials import resolve_ark_api_key

ARK_IMAGE_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
SEEDREAM_LITE_REPAIR_MODEL = "doubao-seedream-5-0-260128"
# Retain the old public name for saved configurations and API payload helpers.
SEEDREAM_STANDARD_REPAIR_MODEL = SEEDREAM_LITE_REPAIR_MODEL
DEFAULT_SEEDREAM_REPAIR_MODEL = SEEDREAM_LITE_REPAIR_MODEL
SEEDREAM_REPAIR_MODELS = (
    DEFAULT_SEEDREAM_REPAIR_MODEL,
    "doubao-seedream-5-0-pro-260628",
)
# Seedream 5.0 Lite accepts fourteen input images; Seedream 5.0 Pro accepts ten.
# The Viewer deliberately sends only two: one complete side reference first,
# followed by the damaged main canvas.  This simple ordering is the one that
# preserved the unusual underside camera on the plush-toy regression job.
SEEDREAM_MAX_INPUT_IMAGES = {
    SEEDREAM_STANDARD_REPAIR_MODEL: 14,
    "doubao-seedream-5-0-pro-260628": 10,
}
# Ark rejects smaller custom outputs for this model.  Keeping the generated
# canvas just over this floor reduced the plush-toy request from 39.5 s to
# 24.4 s in a same-input comparison while retaining the source aspect ratio.
SEEDREAM_MIN_OUTPUT_PIXELS = 3_686_400

# These limits are deliberately below Pillow's decompression-bomb threshold.
# A 4096 x 4096 RGB image already expands to roughly 192 MiB as float32, which
# is ample for current repair providers and still puts a finite bound on a
# malformed or compromised response.
DEFAULT_MAX_IMAGE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 4096 * 4096
DEFAULT_MAX_IMAGE_DIMENSION = 8192
MAX_IMAGE_REDIRECTS = 5
QWEN_IMAGE_MAX_INPUT_BYTES = 10 * 1024 * 1024
QWEN_IMAGE_MAX_INPUT_DIMENSION = 2048

# 提示词写法的两条实测教训：
#   1. 宽泛的关键词否定容易适得其反；禁止项只描述已经实际发生过的具体失败
#      行为（换视角、重画完整对象、保留破损噪声、改动完好区域）。
#   2. 必须给空间锚点。只说"缺失的部分"，模型不知道洞在哪，画出来的圈足
#      明显偏心；说清"底部"之后位置才对得上。
# 所以默认提示词刻意保持简短、正面、并留出物体名称和缺损位置由用户填写。
REFERENCE_PROMPT = (
    "图1是同一对象的一张完整侧视参考图，只用于确认对象的类别、整体结构、部件连接、材质、"
    "颜色和纹理。图2是唯一需要编辑并作为最终结果输出的破损主视图。图2中明显空洞、破碎、"
    "半透明或带有异常彩色闪烁噪声的大片区域，就是对象缺失的待补表面。请依据图1所展示的"
    "真实对象身份和材质，将图2的这片异常破损区域完整替换为对象原本应有的连续表面，使其形状、"
    "体积、材质、颜色、纹理、绒毛或硬表面特征、光照和阴影与图2周围完好区域自然衔接。"
    "最终输出构图只继承图2：保持图2原有的观察方向（包括底视、背视或端面视角）、相机透视、"
    "对象姿态、尺度、位置、外轮廓、未损坏区域、画幅和背景，输出一张与图2同视角的完整修复图。"
    "禁止事项：严禁继承图1的相机视角或构图；严禁生成正面照、侧面照、商品展示照或重新绘制"
    "一个完整的新对象；严禁改变图2的观察方向、姿态、外轮廓、位置、比例和背景；严禁修改图2"
    "中已经完好的区域；严禁把异常彩色闪烁碎片、半透明噪点或破碎纹理保留、复制或扩散到修复面；"
    "严禁输出拼图、多视图、边框、说明文字或多张图片。"
)
DEFAULT_PROMPT = REFERENCE_PROMPT


def minimum_seedream_output_size(
    width: int,
    height: int,
    *,
    minimum_pixels: int = SEEDREAM_MIN_OUTPUT_PIXELS,
    multiple: int = 16,
) -> str:
    """Smallest accepted custom canvas near the source aspect ratio.

    Seedream 5.0 rejects the ``1K`` alias and custom canvases below
    3,686,400 pixels.  Rounding both dimensions upward to a multiple of 16
    avoids silently crossing that floor and keeps projection/resampling close
    to the damaged main view's own aspect ratio.
    """

    width, height = int(width), int(height)
    minimum_pixels, multiple = int(minimum_pixels), int(multiple)
    if width < 1 or height < 1:
        raise ValueError("source image dimensions must be positive")
    if minimum_pixels < 1 or multiple < 1:
        raise ValueError("minimum_pixels and multiple must be positive")
    aspect = float(width) / float(height)
    if not 1.0 / 3.0 <= aspect <= 3.0:
        raise ValueError("Seedream output aspect ratio must be between 1:3 and 3:1")
    target_height = np.sqrt(float(minimum_pixels) / aspect)
    target_width = aspect * target_height
    output_width = int(np.ceil(target_width / multiple) * multiple)
    output_height = int(np.ceil(target_height / multiple) * multiple)
    if output_width * output_height < minimum_pixels:
        output_height += multiple
    return f"{output_width}x{output_height}"


def reference_view_angles(
    count: int,
    damage_elevation: float,
    damage_azimuth: float = 0.0,
    *,
    offset_degrees: float = 28.0,
) -> list:
    """(仰角, 方位角) 列表：在缺损视角**附近**补拍参考图。

    参考图要让模型看清同一块表面的材质与纹样，所以必须是缺损视角的邻近视角，
    绕着它偏开一个锥角即可。

    早先的写法是 ``仰角 = 缺损仰角 × 0.35``。茶壶的缺损仰角是 89 度，乘完得
    31 度侧视，看着合理；换成平铺地毯（缺损仰角 0 度）就彻底失效——两张参考
    的仰角都是 0，只剩方位角不同，第二张 180 度直接翻到了背面，拍出来是糊的。
    绝对角度的缩放没有意义，偏移才有。
    """

    count = int(count)
    if count < 1:
        raise ValueError("参考视角数量必须为正")
    if not 0.0 < float(offset_degrees) < 90.0:
        raise ValueError("offset_degrees 必须在 (0, 90) 之间")

    offset = np.radians(float(offset_degrees))

    def to_direction(elevation, azimuth):
        return np.array([np.cos(elevation) * np.sin(azimuth),
                         np.sin(elevation),
                         np.cos(elevation) * np.cos(azimuth)])

    # 在**方向空间**里绕锥偏移，而不是加减仰角。直接加仰角在两极会被钳住——
    # 茶壶快照仰角 89 度，+28 度钳回 89 度，参考图就又和快照重合了。
    base = to_direction(float(damage_elevation), float(damage_azimuth))
    seed = np.array([0.0, 1.0, 0.0])
    if abs(float(base @ seed)) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(base, seed); u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(base, u_axis)

    angles = []
    for index in range(count):
        roll = 2.0 * np.pi * index / count
        tilted = (np.cos(offset) * base
                  + np.sin(offset) * (np.cos(roll) * u_axis + np.sin(roll) * v_axis))
        tilted /= np.linalg.norm(tilted)
        angles.append((float(np.arcsin(np.clip(tilted[1], -1.0, 1.0))),
                       float(np.arctan2(tilted[0], tilted[2]))))
    return angles


@dataclass(frozen=True)
class RepairConfig:
    """How the repair request is formed."""

    model: str = DEFAULT_SEEDREAM_REPAIR_MODEL
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = ""
    watermark: bool = False
    candidates: int = 1
    timeout_seconds: float = 180.0
    output_size: str = "2K"
    prompt_extend: bool = False
    enable_thinking: bool = False
    # A general I2I model may redraw the object from a reference view.  Viewer
    # calls enable this gate and reject such output before it can be textured.
    enforce_view_consistency: bool = False
    maximum_view_rotation_degrees: float = 5.0
    # Default-model input limit.  Provider-specific builders enforce the
    # tighter per-model value in SEEDREAM_MAX_INPUT_IMAGES.
    maximum_images: int = 14

    def validate(self) -> None:
        if not str(self.model).strip():
            raise ValueError("model must be a non-empty name")
        if not str(self.prompt).strip():
            raise ValueError("prompt must not be empty")
        if int(self.candidates) < 1:
            raise ValueError("candidates must be positive")
        if int(self.candidates) > 6:
            raise ValueError("image repair candidates cannot exceed 6")
        if int(self.maximum_images) < 1:
            raise ValueError("maximum_images 必须为正")
        if float(self.timeout_seconds) <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        output_size = str(self.output_size).strip()
        if not re.fullmatch(r"(?:[234][kK]|[1-9][0-9]*x[1-9][0-9]*)", output_size):
            raise ValueError(
                "output_size must be 2K/3K/4K or an explicit WIDTHxHEIGHT"
            )
        if float(self.maximum_view_rotation_degrees) <= 0.0:
            raise ValueError("maximum_view_rotation_degrees must be positive")


def encode_data_uri(image: np.ndarray) -> str:
    """PNG data URI for a float RGB image in [0, 1]."""

    from PIL import Image

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if max(array.shape[:2]) > QWEN_IMAGE_MAX_INPUT_DIMENSION:
        raise ValueError(
            "qwen-image input dimensions cannot exceed "
            f"{QWEN_IMAGE_MAX_INPUT_DIMENSION}px"
        )
    if np.issubdtype(array.dtype, np.floating):
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(buffer, format="PNG")
    payload = buffer.getvalue()
    if len(payload) > QWEN_IMAGE_MAX_INPUT_BYTES:
        raise ValueError(
            "qwen-image encoded input exceeds the 10 MiB service limit"
        )
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def blank_region(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    colour: Tuple[float, float, float] = (0.60, 0.60, 0.62),
) -> np.ndarray:
    """把待修区域涂成纯色再送给模型。

    看穿一个没有底的壶时，图里那块区域并不空——它塞满了内壁和盖子的细节，
    和"从上往下看一个开口容器"在视觉上无法区分。实测模型据此画出了一个带
    气孔的壶盖，那是对这张图的合理解读，不是模型不行。涂成纯色之后，那块
    区域才真正读作"这里缺东西"。
    """

    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    mask = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image 必须是 (height, width, 3)")
    if mask.shape != image.shape[:2]:
        raise ValueError("mask 分辨率必须与 image 一致")
    result = image.copy()
    result[mask] = np.asarray(colour, dtype=np.float32)
    return result


def make_edit_mask_guide(mask: np.ndarray) -> np.ndarray:
    """Return the provider-visible black/white edit map for image slot 2.

    The DashScope Qwen Image I2I schema has no separate mask property.  A
    binary RGB image plus an explicit textual role is therefore the only mask
    information the model itself receives.  Keeping it binary also makes the
    exact request easy to audit by eye.
    """

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any() or mask.all():
        raise ValueError("edit mask must be a non-empty proper 2D subset")
    return np.repeat(mask[..., None], 3, axis=2).astype(np.float32)


def make_material_swatch(image: np.ndarray, maximum_size: int = 512) -> np.ndarray:
    """Turn an appearance reference into a square, composition-free swatch."""

    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("material reference must have shape (height, width, 3)")
    height, width = image.shape[:2]
    side = min(height, width, int(maximum_size))
    if side < 32:
        raise ValueError("material reference is too small for a useful swatch")
    top = (height - side) // 2
    left = (width - side) // 2
    return image[top:top + side, left:left + side].copy()


def make_reference_overview(
    reference_rgbs: Sequence[np.ndarray],
    *,
    columns: int = 3,
    gutter_pixels: int = 8,
    maximum_dimension: int = QWEN_IMAGE_MAX_INPUT_DIMENSION,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Place complete reference frames on one uncropped recognition board.

    Ark's ``image`` array does not carry per-image role metadata.  Supplying
    six peers next to one unusual underside view caused Seedream to choose a
    conventional front reference as its output canvas.  One overview keeps all
    six identity observations but restores the successful web-client contract:
    reference context first, unique edit canvas last.
    """

    images = []
    for reference in reference_rgbs:
        image = np.clip(np.asarray(reference, dtype=np.float32), 0.0, 1.0)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("reference images must have shape (height, width, 3)")
        if min(image.shape[:2]) < 1:
            raise ValueError("reference images must not be empty")
        images.append(image)
    if not images:
        raise ValueError("at least one complete reference image is required")
    columns = max(1, min(int(columns), len(images)))
    gutter = max(0, int(gutter_pixels))
    maximum_dimension = int(maximum_dimension)
    if maximum_dimension < 64:
        raise ValueError("maximum_dimension is too small")
    rows = int(np.ceil(len(images) / columns))
    natural_cell_width = max(int(image.shape[1]) for image in images)
    natural_cell_height = max(int(image.shape[0]) for image in images)
    natural_width = columns * natural_cell_width + (columns - 1) * gutter
    natural_height = rows * natural_cell_height + (rows - 1) * gutter
    scale = min(
        1.0,
        maximum_dimension / max(natural_width, 1),
        maximum_dimension / max(natural_height, 1),
    )
    cell_width = max(1, int(np.floor(natural_cell_width * scale)))
    cell_height = max(1, int(np.floor(natural_cell_height * scale)))
    scaled_gutter = max(0, int(round(gutter * scale)))
    board_width = columns * cell_width + (columns - 1) * scaled_gutter
    board_height = rows * cell_height + (rows - 1) * scaled_gutter
    board = np.ones((board_height, board_width, 3), dtype=np.float32)
    placements = []
    import cv2

    for index, image in enumerate(images):
        fit = min(cell_width / image.shape[1], cell_height / image.shape[0])
        target_width = max(1, int(round(image.shape[1] * fit)))
        target_height = max(1, int(round(image.shape[0] * fit)))
        if (target_height, target_width) == image.shape[:2]:
            resized = image
        else:
            interpolation = cv2.INTER_AREA if fit < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(
                image, (target_width, target_height), interpolation=interpolation
            ).astype(np.float32)
        row, column = divmod(index, columns)
        left = column * (cell_width + scaled_gutter) + (cell_width - target_width) // 2
        top = row * (cell_height + scaled_gutter) + (cell_height - target_height) // 2
        board[top:top + target_height, left:left + target_width] = resized
        placements.append(
            {
                "source_index": index + 1,
                "source_size": [int(image.shape[1]), int(image.shape[0])],
                "box_xywh": [left, top, target_width, target_height],
                "cropped": False,
            }
        )
    return board, {
        "layout": f"{columns}x{rows}",
        "source_images": len(images),
        "board_size": [board_width, board_height],
        "gutter_pixels": scaled_gutter,
        "placements": placements,
        "all_sources_uncropped": True,
    }


def prepare_mask_guided_request(
    snapshot_rgb: np.ndarray,
    hole_mask: np.ndarray,
    reference_rgbs: Sequence[np.ndarray],
    config: RepairConfig,
    *,
    blank_hole: bool,
) -> Tuple[List[np.ndarray], List[str], RepairConfig, Dict[str, object]]:
    """Build the Viewer's complete-frame multi-image Seedream request.

    Image 1 is one complete side reference.  Image 2 is the complete damaged
    main view and the only requested output canvas.  This is the two-image
    order that produced the accepted result on the plush-toy regression job.
    Extra references are deliberately ignored: overview boards and peer image
    arrays repeatedly made Seedream redraw a conventional side/front view
    instead of repairing the unusual underside canvas.  The mask is deliberately
    *not* sent to the model; it is
    only used locally after generation to composite the result for texture
    projection.  ``blank_hole`` remains in the signature for compatibility but
    no longer modifies provider input.
    """

    snapshot = np.clip(np.asarray(snapshot_rgb, dtype=np.float32), 0.0, 1.0)
    mask = np.asarray(hole_mask, dtype=bool)
    if snapshot.ndim != 3 or snapshot.shape[2] != 3:
        raise ValueError("snapshot_rgb must have shape (height, width, 3)")
    if mask.shape != snapshot.shape[:2]:
        raise ValueError("hole_mask must match snapshot_rgb")
    if int(config.maximum_images) < 2:
        raise ValueError("multi-view repair requires at least two input images")
    valid_references = []
    for reference in reference_rgbs:
        image = np.clip(np.asarray(reference, dtype=np.float32), 0.0, 1.0)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("reference images must have shape (height, width, 3)")
        valid_references.append(image)
    if not valid_references:
        raise ValueError(
            "multi-view repair requires at least one complete clean reference image"
        )
    selected = valid_references[:1]
    images = [selected[0].copy(), snapshot.copy()]
    roles = ["complete_side_reference", "damaged_main_view"]
    user_instruction = str(config.prompt).strip()
    return images, roles, config, {
        "input_layout": list(roles),
        "reference_images_supplied": int(len(valid_references)),
        "reference_images_used": int(len(selected)),
        "reference_images_dropped": int(len(valid_references) - len(selected)),
        "provider_input_images": 2,
        "reference_policy": "single_complete_side_reference_first",
        # Compatibility keys consumed by old callers and audit readers.
        "material_references_supplied": int(len(valid_references)),
        "material_references_used": 1,
        "material_references_dropped": int(len(valid_references) - 1),
        "mask_pixels": int(mask.sum()),
        "mask_fraction_of_frame": float(mask.mean()),
        "hole_blanked_before_sending": False,
        "mask_encoding": "local_postprocess_only",
        "binary_mask_sent_as_model_input": False,
        "user_prompt": user_instruction,
        "effective_prompt": user_instruction,
    }


def build_payload(
    image_refs: Sequence[str], config: RepairConfig | None = None
) -> Dict[str, object]:
    """The request body: images first, then the instruction.

    Content order is meaningful to these models -- the images are the subject
    and the text is what to do with them -- so the instruction goes last.
    """

    config = config or RepairConfig()
    config.validate()
    refs = [str(ref) for ref in image_refs if str(ref).strip()]
    if not refs:
        raise ValueError("at least one input image is required")
    if len(refs) > int(config.maximum_images):
        raise ValueError(
            f"qwen-image accepts at most {int(config.maximum_images)} input images"
        )
    content: List[Dict[str, str]] = [{"image": ref} for ref in refs]
    content.append({"text": str(config.prompt)})
    parameters: Dict[str, object] = {
        "n": int(config.candidates),
        "watermark": bool(config.watermark),
        "prompt_extend": bool(config.prompt_extend),
        "enable_thinking": bool(config.enable_thinking),
    }
    if str(config.negative_prompt):
        parameters["negative_prompt"] = str(config.negative_prompt)
    return {
        "model": str(config.model),
        "input": {"messages": [{"role": "user", "content": content}]},
        "parameters": parameters,
    }


def build_ark_payload(
    image_refs: Sequence[str], config: RepairConfig | None = None
) -> Dict[str, object]:
    """Build Volcengine Ark's Seedream multi-image editing payload."""

    config = config or RepairConfig(model=DEFAULT_SEEDREAM_REPAIR_MODEL)
    config.validate()
    if str(config.model).strip() not in SEEDREAM_REPAIR_MODELS:
        raise ValueError(
            "Seedream repair model must be one of: "
            + ", ".join(SEEDREAM_REPAIR_MODELS)
        )
    refs = [str(ref) for ref in image_refs if str(ref).strip()]
    provider_maximum = int(SEEDREAM_MAX_INPUT_IMAGES[str(config.model).strip()])
    maximum = min(int(config.maximum_images), provider_maximum)
    if not refs:
        raise ValueError("Seedream repair requires at least one input image")
    if len(refs) > maximum:
        raise ValueError(
            f"Seedream repair accepts at most {maximum} complete input images "
            "when generating one output"
        )
    if int(config.candidates) != 1:
        raise ValueError("Seedream single-image repair requires candidates=1")
    payload = {
        "model": str(config.model),
        "prompt": str(config.prompt),
        "image": refs,
        "response_format": "url",
        "size": str(config.output_size).strip(),
        "stream": False,
        "watermark": bool(config.watermark),
    }
    # The standard/Lite route supports group generation.  Pro is single-output
    # and rejects this Lite-only switch, so omitting it is part of the model
    # adapter rather than a cosmetic payload difference.
    if str(config.model).strip() == SEEDREAM_STANDARD_REPAIR_MODEL:
        payload["sequential_image_generation"] = "disabled"
    return payload


def extract_image_urls(response: Dict[str, object]) -> List[str]:
    """Pull image URLs out of a response.

    Written tolerantly on purpose: the exact envelope is version-dependent, and
    a parser that only accepts one shape turns a working call into a silent
    failure.  Anything that looks like an image reference is collected.
    """

    found: List[str] = []

    def visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("image", "url", "image_url") and isinstance(value, str):
                    if value.startswith("http") or value.startswith("data:image"):
                        found.append(value)
                else:
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(response)
    seen, unique = set(), []
    for url in found:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def extract_qwen_image_urls(response: Dict[str, object]) -> List[str]:
    """Parse the Qwen-Image synchronous response envelope.

    Qwen Image 3.0 returns all ``n`` images in one choice on some deployments,
    while older examples show one image per choice.  The content items may also
    use ``image_url`` instead of ``image``.  Choice/message validation remains
    strict, but output grouping is not part of the repair contract.
    """

    output = response.get("output")
    choices = output.get("choices") if isinstance(output, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("qwen-image response has no output.choices")
    urls: List[str] = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            raise RuntimeError(f"qwen-image choice {index} is malformed")
        finish_reason = choice.get("finish_reason")
        if finish_reason not in (None, "stop"):
            raise RuntimeError(
                f"qwen-image choice {index} did not finish successfully: "
                f"{finish_reason!r}"
            )
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") not in (
            None,
            "assistant",
        ):
            raise RuntimeError(f"qwen-image choice {index} has no assistant message")
        content = message.get("content")
        if not isinstance(content, list):
            raise RuntimeError(f"qwen-image choice {index} has malformed content")
        for item in content:
            if not isinstance(item, dict):
                continue
            for key in ("image", "image_url", "url"):
                value = item.get(key)
                if isinstance(value, str) and (
                    value.startswith("https://") or value.startswith("data:image")
                ):
                    urls.append(value)
                    break
    unique = list(dict.fromkeys(urls))
    if not unique:
        raise RuntimeError("qwen-image response choices contain no image")
    return unique


def _trusted_ark_image_endpoint(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").rstrip(".").lower()
        != "ark.cn-beijing.volces.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/api/v3/images/generations"
    ):
        raise ValueError("refusing to send an Ark API key to an untrusted endpoint")
    return str(value).strip()


def _safe_service_error(response, *, api_key: str | None = None) -> str:
    try:
        body = response.json()
    except Exception:
        body = {}
    fields = []
    if isinstance(body, dict):
        # Ark places failures under "error"; some services use top-level fields.
        nested = body.get("error")
        sources = [nested, body] if isinstance(nested, dict) else [body]
        for key in ("code", "message", "request_id"):
            for source in sources:
                value = source.get(key)
                if isinstance(value, (str, int)) and value != "":
                    fields.append(f"{key}={value}")
                    break
    message = " ".join(fields) or "service returned an error"
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(
        r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", message
    )
    message = re.sub(
        r"(?i)(ARK_API_KEY\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?i)(DASHSCOPE_API_KEY\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    message = re.sub(
        r"(?<![A-Za-z0-9])sk-[A-Za-z0-9._~+/=-]{8,}",
        "[REDACTED]",
        message,
        flags=re.IGNORECASE,
    )
    message = re.sub(
        r"(?<![A-Za-z0-9])ark-[A-Za-z0-9._~+/=-]{8,}",
        "[REDACTED]",
        message,
        flags=re.IGNORECASE,
    )
    return message[:400]


def request_repair(
    image_refs: Sequence[str],
    *,
    api_key: str | None = None,
    config: RepairConfig | None = None,
    url: str = ARK_IMAGE_URL,
    session=None,
) -> Tuple[List[str], Dict[str, object]]:
    """POST the repair request and return the produced image URLs."""

    import requests

    config = config or RepairConfig()
    caller = session or requests
    model = str(config.model).strip()
    if model.lower().startswith("doubao-") and model not in SEEDREAM_REPAIR_MODELS:
        raise ValueError(
            "unsupported Seedream repair model: " + model
        )
    is_ark = model in SEEDREAM_REPAIR_MODELS
    if not is_ark:
        raise ValueError(
            "Qwen image API execution is retired; use "
            f"{DEFAULT_SEEDREAM_REPAIR_MODEL} through Volcengine Ark"
        )
    key = resolve_ark_api_key(api_key)
    payload = build_ark_payload(image_refs, config)
    endpoint = _trusted_ark_image_endpoint(url)
    response = caller.post(
        endpoint,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=float(config.timeout_seconds),
        allow_redirects=False,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"image repair failed: HTTP {response.status_code} "
            f"{_safe_service_error(response, api_key=key)}"
        )
    body = response.json()
    urls = extract_image_urls(body)
    if len(urls) != int(config.candidates):
        raise RuntimeError(
            f"image service returned {len(urls)} images for "
            f"candidates={int(config.candidates)}"
        )
    return urls, body


def fetch_image(
    url: str,
    *,
    timeout: float = 120.0,
    session=None,
    max_download_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
) -> np.ndarray:
    """Download (or decode) a bounded produced image as float RGB in [0, 1].

    Remote images must use HTTPS.  Literal loopback/private/link-local IPs and
    ``localhost`` are rejected before the request; hostnames are intentionally
    not resolved here so test sessions and signed CDN URLs remain usable.  The
    streamed body, decoded pixel count, and each image dimension are bounded.
    """

    from PIL import Image

    byte_limit = int(max_download_bytes)
    pixel_limit = int(max_pixels)
    dimension_limit = int(max_dimension)
    if byte_limit <= 0 or pixel_limit <= 0 or dimension_limit <= 0:
        raise ValueError("image download and dimension limits must be positive")

    value = str(url).strip()
    if value.startswith("data:image"):
        if "," not in value or ";base64," not in value[: value.index(",") + 1]:
            raise ValueError("image data URI must contain a base64 payload")
        payload = value.split(",", 1)[1]
        # Reject before allocating the decoded copy.  Four base64 characters
        # encode at most three bytes; the small allowance covers terminal '='.
        maximum_encoded = 4 * ((byte_limit + 2) // 3) + 4
        if len(payload) > maximum_encoded:
            raise RuntimeError(
                f"produced image exceeds the {byte_limit}-byte download limit"
            )
        try:
            data = base64.b64decode(payload, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("image data URI has invalid base64") from exc
        if len(data) > byte_limit:
            raise RuntimeError(
                f"produced image exceeds the {byte_limit}-byte download limit"
            )
    else:
        import requests

        caller = session or requests
        response = None
        current_url = value
        for redirect_count in range(MAX_IMAGE_REDIRECTS + 1):
            parsed = urlsplit(current_url)
            if parsed.scheme.lower() != "https" or not parsed.hostname:
                raise ValueError("remote produced images must use an absolute HTTPS URL")
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("remote image URLs must not contain credentials")
            hostname = parsed.hostname.rstrip(".").lower()
            if hostname == "localhost" or hostname.endswith(".localhost"):
                raise ValueError("remote image URL points to a local host")
            try:
                address = ipaddress.ip_address(hostname)
            except ValueError:
                address = None
            if address is not None and not address.is_global:
                raise ValueError("remote image URL points to a non-public IP address")

            try:
                response = caller.get(
                    current_url,
                    timeout=float(timeout),
                    stream=True,
                    allow_redirects=False,
                )
            except TypeError:
                # Preserve compatibility with deliberately tiny injected test
                # sessions.  Their already-materialized ``content`` is still
                # checked below.
                response = caller.get(current_url, timeout=float(timeout))
            if response.status_code not in (301, 302, 303, 307, 308):
                break
            headers = getattr(response, "headers", {}) or {}
            location = headers.get("Location") or headers.get("location")
            close = getattr(response, "close", None)
            if callable(close):
                close()
            response = None
            if not location:
                raise RuntimeError("image download redirect has no Location header")
            if redirect_count >= MAX_IMAGE_REDIRECTS:
                raise RuntimeError("image download exceeded the redirect limit")
            # Validate the resolved target at the top of the next iteration,
            # before any request can reach it.
            current_url = urljoin(current_url, str(location))
        if response is None:  # defensive: the loop either breaks or raises
            raise RuntimeError("image download produced no response")
        try:
            if response.status_code != 200:
                raise RuntimeError(
                    f"could not download the image: HTTP {response.status_code}"
                )
            headers = getattr(response, "headers", {}) or {}
            content_length = headers.get("Content-Length") or headers.get("content-length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except (TypeError, ValueError):
                    declared = None
                if declared is not None and declared > byte_limit:
                    raise RuntimeError(
                        f"produced image exceeds the {byte_limit}-byte download limit"
                    )

            iterator = getattr(response, "iter_content", None)
            if callable(iterator):
                chunks = []
                size = 0
                for chunk in iterator(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > byte_limit:
                        raise RuntimeError(
                            f"produced image exceeds the {byte_limit}-byte download limit"
                        )
                    chunks.append(chunk)
                data = b"".join(chunks)
            else:
                data = response.content
                if len(data) > byte_limit:
                    raise RuntimeError(
                        f"produced image exceeds the {byte_limit}-byte download limit"
                    )
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    with Image.open(io.BytesIO(data)) as encoded:
        width, height = encoded.size
        if width <= 0 or height <= 0:
            raise ValueError("produced image has invalid dimensions")
        if width > dimension_limit or height > dimension_limit:
            raise RuntimeError(
                "produced image exceeds the "
                f"{dimension_limit}-pixel dimension limit: {width}x{height}"
            )
        if width * height > pixel_limit:
            raise RuntimeError(
                f"produced image exceeds the {pixel_limit}-pixel limit: {width}x{height}"
            )
        image = encoded.convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def align_snapshot_repair(
    original: np.ndarray,
    repaired: np.ndarray,
    mask: np.ndarray,
    *,
    maximum_rotation_degrees: float = 5.0,
    minimum_matches: int = 32,
    minimum_inlier_fraction: float = 0.25,
    maximum_output_dimension: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Register a generated *same-snapshot* repair without orientation flips.

    A generated repair is not a separately photographed object: its orientation
    is already authoritative because the first model input was the frozen
    snapshot.  Silhouette-only registration is therefore the wrong tool.  A
    rug, plate or tabletop can have an almost 180-degree-symmetric outline, and
    maximising silhouette IoU may turn a visually upright repair upside down.

    This function matches local image content outside the physical hole,
    estimates a conservative similarity transform with RANSAC, rejects flips
    and rotations larger than ``maximum_rotation_degrees``, then creates an
    aligned canvas at close to the model output's native resolution.  The
    original snapshot and mask are scaled to that same canvas, so downstream
    texture sampling does not first throw the generated high-resolution pixels
    away.
    """

    import cv2
    from scipy.ndimage import binary_dilation, gaussian_filter

    original = np.clip(np.asarray(original, dtype=np.float32), 0.0, 1.0)
    repaired = np.clip(np.asarray(repaired, dtype=np.float32), 0.0, 1.0)
    mask = np.asarray(mask, dtype=bool)
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("original must have shape (height, width, 3)")
    if repaired.ndim != 3 or repaired.shape[2] != 3:
        raise ValueError("repaired must have shape (height, width, 3)")
    if mask.shape != original.shape[:2] or not mask.any() or mask.all():
        raise ValueError("mask must be a non-empty proper subset of original")
    if float(maximum_rotation_degrees) <= 0.0:
        raise ValueError("maximum_rotation_degrees must be positive")

    target_height, target_width = original.shape[:2]
    source_height, source_width = repaired.shape[:2]
    # Feature extraction at more than 2K is wasteful for a global alignment and
    # can make a 4K model response unexpectedly expensive.  Coordinates are
    # mapped back to the original image domains before fitting the transform.
    feature_limit = 2048.0
    source_feature_scale = min(1.0, feature_limit / max(source_height, source_width))
    target_feature_scale = min(1.0, feature_limit / max(target_height, target_width))

    def feature_image(image, scale):
        rgb = np.rint(image * 255.0).astype(np.uint8)
        if scale < 1.0:
            rgb = cv2.resize(
                rgb,
                (int(round(rgb.shape[1] * scale)), int(round(rgb.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    source_gray = feature_image(repaired, source_feature_scale)
    target_gray = feature_image(original, target_feature_scale)
    safe_context = ~binary_dilation(
        mask,
        iterations=max(8, int(round(min(target_height, target_width) * 0.02))),
    )
    if target_feature_scale < 1.0:
        safe_context = cv2.resize(
            safe_context.astype(np.uint8),
            (target_gray.shape[1], target_gray.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    feature_mask = safe_context.astype(np.uint8) * 255

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(
            nfeatures=12000, contrastThreshold=0.01, edgeThreshold=20
        )
        norm = cv2.NORM_L2
        ratio_limit = 0.72
    else:  # pragma: no cover - current OpenCV wheels include SIFT
        detector = cv2.ORB_create(nfeatures=16000, fastThreshold=5)
        norm = cv2.NORM_HAMMING
        ratio_limit = 0.78
    source_keypoints, source_descriptors = detector.detectAndCompute(source_gray, None)
    target_keypoints, target_descriptors = detector.detectAndCompute(
        target_gray, feature_mask
    )
    if source_descriptors is None or target_descriptors is None:
        raise RuntimeError("修复图配准失败：洞外没有足够的可匹配纹理")
    pairs = cv2.BFMatcher(norm).knnMatch(source_descriptors, target_descriptors, k=2)
    matches = [first for first, second in pairs if first.distance < ratio_limit * second.distance]
    if len(matches) < int(minimum_matches):
        raise RuntimeError(
            f"修复图配准失败：洞外仅有 {len(matches)} 个可靠特征匹配"
        )
    source_points = np.float32(
        [source_keypoints[item.queryIdx].pt for item in matches]
    ) / float(source_feature_scale)
    target_points = np.float32(
        [target_keypoints[item.trainIdx].pt for item in matches]
    ) / float(target_feature_scale)
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=10000,
        confidence=0.999,
        refineIters=20,
    )
    if affine is None or inlier_mask is None:
        raise RuntimeError("修复图配准失败：RANSAC无法确定同向变换")
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(inliers.sum())
    inlier_fraction = inlier_count / max(len(matches), 1)
    if inlier_count < int(minimum_matches) or inlier_fraction < float(minimum_inlier_fraction):
        raise RuntimeError(
            "修复图配准失败：洞外匹配缺少一致性 "
            f"({inlier_count}/{len(matches)}, {inlier_fraction:.1%})"
        )
    linear = np.asarray(affine[:, :2], dtype=np.float64)
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(affine).all() or determinant <= 0.0:
        raise RuntimeError("修复图配准拒绝了翻转或退化变换")
    scale = float(np.sqrt(determinant))
    rotation_degrees = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    if abs(rotation_degrees) > float(maximum_rotation_degrees):
        raise RuntimeError(
            "修复图方向与快照不一致："
            f"估计旋转 {rotation_degrees:.2f}°，允许范围为"
            f" ±{float(maximum_rotation_degrees):.1f}°"
        )
    if not 0.2 <= scale <= 5.0:
        raise RuntimeError(f"修复图配准比例异常：{scale:.4f}")

    predicted = source_points[inliers] @ linear.T + np.asarray(affine[:, 2])
    reprojection = np.linalg.norm(predicted - target_points[inliers], axis=1)
    median_reprojection = float(np.median(reprojection))
    p90_reprojection = float(np.percentile(reprojection, 90.0))
    if p90_reprojection > 5.0:
        raise RuntimeError(
            f"修复图配准残差过大：p90={p90_reprojection:.2f}px"
        )

    homography = np.eye(3, dtype=np.float64)
    homography[:2] = affine
    snapshot_warp = cv2.warpPerspective(
        repaired,
        homography,
        (target_width, target_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)
    valid_context = ~binary_dilation(
        mask,
        iterations=max(8, int(round(min(target_height, target_width) * 0.02))),
    )
    original_low = gaussian_filter(original, sigma=(4.0, 4.0, 0.0))
    warped_low = gaussian_filter(snapshot_warp, sigma=(4.0, 4.0, 0.0))
    context_mae = float(np.mean(np.abs(warped_low[valid_context] - original_low[valid_context])))
    original_luma = original_low[valid_context].mean(axis=1)
    warped_luma = warped_low[valid_context].mean(axis=1)
    context_correlation = float(np.corrcoef(original_luma, warped_luma)[0, 1])
    if not np.isfinite(context_correlation) or context_correlation < 0.25:
        raise RuntimeError(
            "修复图配准后的洞外内容仍不一致："
            f"correlation={context_correlation:.3f}"
        )

    # Use the largest uniform canvas that does not upsample either source
    # dimension.  Multiplying source->snapshot H by this scale maps the raw
    # repair almost one-to-one into the aligned texture canvas.
    native_scale = max(
        1.0,
        min(source_width / float(target_width), source_height / float(target_height)),
    )
    native_scale = min(
        native_scale,
        float(maximum_output_dimension) / max(target_width, target_height),
    )
    output_width = int(round(target_width * native_scale))
    output_height = int(round(target_height * native_scale))
    canvas_scale_x = output_width / float(target_width)
    canvas_scale_y = output_height / float(target_height)
    if abs(canvas_scale_x - canvas_scale_y) > 0.002:
        # Integer dimensions can differ by a fraction of a pixel.  Camera code
        # supports independent fx/fy, so record both rather than silently
        # distorting the repair.
        pass
    canvas_transform = np.diag([canvas_scale_x, canvas_scale_y, 1.0]) @ homography
    aligned_repair = cv2.warpPerspective(
        repaired,
        canvas_transform,
        (output_width, output_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)
    source_valid = cv2.warpPerspective(
        np.ones((source_height, source_width), dtype=np.uint8),
        canvas_transform,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    aligned_original = cv2.resize(
        original,
        (output_width, output_height),
        interpolation=cv2.INTER_LANCZOS4,
    ).astype(np.float32)
    aligned_mask = cv2.resize(
        mask.astype(np.uint8),
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    hole_valid_fraction = float(np.mean(source_valid[aligned_mask]))
    if hole_valid_fraction < 0.995:
        raise RuntimeError(
            "修复图没有完整覆盖物理洞："
            f"coverage={hole_valid_fraction:.2%}"
        )
    aligned_repair[~source_valid] = aligned_original[~source_valid]
    return (
        np.clip(aligned_original, 0.0, 1.0),
        np.clip(aligned_repair, 0.0, 1.0),
        aligned_mask,
        {
            "pipeline": "same_snapshot_feature_alignment_v1",
            "mode": "content_similarity_native_canvas",
            "source_size": [int(source_width), int(source_height)],
            "snapshot_size": [int(target_width), int(target_height)],
            "output_size": [int(output_width), int(output_height)],
            "canvas_scale_x": float(canvas_scale_x),
            "canvas_scale_y": float(canvas_scale_y),
            "rotation_degrees": rotation_degrees,
            "source_to_snapshot_scale": scale,
            "matches": int(len(matches)),
            "inliers": inlier_count,
            "inlier_fraction": float(inlier_fraction),
            "median_reprojection_error_pixels": median_reprojection,
            "p90_reprojection_error_pixels": p90_reprojection,
            "context_low_frequency_mae": context_mae,
            "context_correlation": context_correlation,
            "hole_source_coverage": hole_valid_fraction,
            "source_to_snapshot_matrix": homography.tolist(),
            "source_to_canvas_matrix": canvas_transform.tolist(),
            "orientation_flip_allowed": False,
        },
    )


def multiband_composite_repair(
    original: np.ndarray,
    repaired: np.ndarray,
    mask: np.ndarray,
    *,
    transition_pixels: float = 8.0,
    low_frequency_sigma: float = 2.0,
    low_frequency_transition_multiplier: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Join an aligned repair to exact outside pixels without a hard sticker edge.

    Outside ``mask`` remains byte-for-byte equal to ``original``.  Inside a
    narrow seam band, healthy boundary pixels are extended along the nearest
    contour normal and blended into the repair.  High frequencies transition
    over ``transition_pixels``; low-frequency tone transitions more gradually.
    This avoids mixing the see-through/deleted content from deep inside the
    physical hole while still enforcing C0 appearance at its real edge.
    """

    from scipy.ndimage import distance_transform_edt, gaussian_filter

    original = np.clip(np.asarray(original, dtype=np.float32), 0.0, 1.0)
    repaired = np.clip(np.asarray(repaired, dtype=np.float32), 0.0, 1.0)
    mask = np.asarray(mask, dtype=bool)
    if original.shape != repaired.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("original and repaired must have the same (H, W, 3) shape")
    if mask.shape != original.shape[:2] or not mask.any() or mask.all():
        raise ValueError("mask must be a non-empty proper subset of the image")
    transition = float(transition_pixels)
    sigma = float(low_frequency_sigma)
    multiplier = float(low_frequency_transition_multiplier)
    if transition <= 0.0 or sigma <= 0.0 or multiplier < 1.0:
        raise ValueError("transition/sigma must be positive and multiplier >= 1")

    distance, nearest = distance_transform_edt(mask, return_indices=True)
    context_extension = original[nearest[0], nearest[1]]
    repair_low = gaussian_filter(repaired, sigma=(sigma, sigma, 0.0))
    context_low = gaussian_filter(context_extension, sigma=(sigma, sigma, 0.0))
    repair_high = repaired - repair_low
    context_high = context_extension - context_low

    def smoothstep(value):
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    high_weight = smoothstep(distance / transition)
    low_weight = smoothstep(distance / (transition * multiplier))
    blended = (
        context_low * (1.0 - low_weight[..., None])
        + repair_low * low_weight[..., None]
        + context_high * (1.0 - high_weight[..., None])
        + repair_high * high_weight[..., None]
    )
    result = original.copy()
    result[mask] = np.clip(blended[mask], 0.0, 1.0)

    seam_values = []
    for axis in (0, 1):
        first_mask = np.take(mask, range(mask.shape[axis] - 1), axis=axis)
        second_mask = np.take(mask, range(1, mask.shape[axis]), axis=axis)
        crossing = first_mask != second_mask
        first = np.take(result, range(result.shape[axis] - 1), axis=axis)
        second = np.take(result, range(1, result.shape[axis]), axis=axis)
        seam_values.append(np.mean(np.abs(first - second), axis=2)[crossing])
    seam_values = np.concatenate(seam_values)
    return result.astype(np.float32), {
        "pipeline": "inward_multiband_composite_v1",
        "mask_pixels": int(mask.sum()),
        "transition_pixels": transition,
        "low_frequency_sigma_pixels": sigma,
        "low_frequency_transition_pixels": transition * multiplier,
        "outside_mask_untouched": bool(np.array_equal(result[~mask], original[~mask])),
        "seam_l1_rgb_per_channel_median_255": float(np.median(seam_values) * 255.0),
        "seam_l1_rgb_per_channel_p90_255": float(np.percentile(seam_values, 90.0) * 255.0),
        "seam_l1_rgb_per_channel_p95_255": float(np.percentile(seam_values, 95.0) * 255.0),
    }


def composite_repair(
    original: np.ndarray,
    repaired: np.ndarray,
    mask: np.ndarray,
    *,
    feather_pixels: int = 3,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Take the model's pixels inside ``mask`` only, keeping the rest exact.

    The model is free to redraw the whole frame; this constrains it to the one
    region that was genuinely empty.  ``repaired`` is resized to the original's
    resolution first, since these models return their own size.
    """

    from PIL import Image

    original = np.clip(np.asarray(original, dtype=np.float32), 0.0, 1.0)
    repaired = np.clip(np.asarray(repaired, dtype=np.float32), 0.0, 1.0)
    mask = np.asarray(mask, dtype=bool)
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("original must have shape (height, width, 3)")
    if mask.shape != original.shape[:2]:
        raise ValueError("mask must match the original's resolution")
    if not mask.any():
        raise ValueError("mask is empty; nothing would be taken from the repair")

    height, width = original.shape[:2]
    resized = False
    if repaired.shape[:2] != (height, width):
        image = Image.fromarray(np.rint(repaired * 255.0).astype(np.uint8))
        repaired = np.asarray(
            image.resize((width, height), Image.LANCZOS)
        ).astype(np.float32) / 255.0
        resized = True

    weight = mask.astype(np.float32)
    if int(feather_pixels) > 0:
        from scipy.ndimage import gaussian_filter

        weight = gaussian_filter(weight, sigma=float(feather_pixels))
        weight = np.clip(weight, 0.0, 1.0)
        weight[mask] = 1.0          # never dilute the interior of the hole
    result = original * (1.0 - weight[..., None]) + repaired * weight[..., None]
    result = np.clip(result, 0.0, 1.0)

    def seam_values(image):
        values = []
        for axis in (0, 1):
            first_mask = np.take(mask, range(mask.shape[axis] - 1), axis=axis)
            second_mask = np.take(mask, range(1, mask.shape[axis]), axis=axis)
            crossing = first_mask != second_mask
            first = np.take(image, range(image.shape[axis] - 1), axis=axis)
            second = np.take(image, range(1, image.shape[axis]), axis=axis)
            values.append(np.mean(np.abs(first - second), axis=2)[crossing])
        return np.concatenate(values)

    raw_seam = seam_values(result)
    from scipy.ndimage import gaussian_filter

    low_frequency_seam = seam_values(
        gaussian_filter(result, sigma=(2.0, 2.0, 0.0))
    )
    return result, {
        "pipeline": "masked_composite_v1",
        "mask_pixels": int(mask.sum()),
        "repaired_resized": resized,
        "feather_pixels": int(feather_pixels),
        "outside_mask_untouched": bool(int(feather_pixels) == 0),
        "seam_raw_l1_rgb_per_channel_median_255": float(
            np.median(raw_seam) * 255.0
        ),
        "seam_raw_l1_rgb_per_channel_p90_255": float(
            np.percentile(raw_seam, 90.0) * 255.0
        ),
        "seam_low_frequency_sigma_pixels": 2.0,
        "seam_low_frequency_l1_rgb_per_channel_median_255": float(
            np.median(low_frequency_seam) * 255.0
        ),
        "seam_low_frequency_l1_rgb_per_channel_p90_255": float(
            np.percentile(low_frequency_seam, 90.0) * 255.0
        ),
    }


def harmonize_repair_to_context(
    original: np.ndarray,
    repaired: np.ndarray,
    mask: np.ndarray,
    *,
    inner_ring_pixels: int = 8,
    outer_ring_pixels: int = 48,
    lowpass_sigma: float = 8.0,
    maximum_blur_sigma: float = 3.0,
    downstream_blur_sigma: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Match an aligned repair's colour and sharpness to the local context.

    Image generators often redraw and sharpen the whole frame even when asked
    only to fill a hole.  The original image is invalid *inside* ``mask``, but
    the two aligned images are both observable in a safe annulus outside it.
    This function uses that annulus to estimate a robust low-frequency affine
    colour mapping and, only when the repair is measurably sharper, a small
    anti-aliasing blur.  The mapping is then applied to the repair; no pixel
    from the damaged original is mixed into the hole.

    The function deliberately fails open.  If the annulus is too small or has
    too little variation to identify a mapping, the repair is returned
    unchanged and ``applied`` is false.  That makes it safe for featureless
    objects and for synthetic unit-test images.

    ``downstream_blur_sigma`` is the width, in pixels of *this* image, of a
    low-pass the caller will apply afterwards -- for the viewer, the footprint
    of the Gaussian that will splat each sampled colour into 3D.  The thing
    that has to match the context is the final render, not the intermediate
    image, so that stage's blur is folded into the sharpness comparison and
    only the excess it will *not* remove is taken out here.  Ignoring it blurs
    twice: on a carpet job the patch landed at 0.53x the surrounding carpet's
    high-frequency energy, where accounting for it gives 0.86x.
    """

    from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

    original = np.clip(np.asarray(original, dtype=np.float32), 0.0, 1.0)
    repaired = np.clip(np.asarray(repaired, dtype=np.float32), 0.0, 1.0)
    mask = np.asarray(mask, dtype=bool)
    if original.shape != repaired.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("original and repaired must have the same (H, W, 3) shape")
    if mask.shape != original.shape[:2]:
        raise ValueError("mask must match the image resolution")
    if not mask.any():
        raise ValueError("mask is empty")
    inner = max(0, int(inner_ring_pixels))
    outer = max(inner + 1, int(outer_ring_pixels))
    inner_mask = binary_dilation(mask, iterations=inner) if inner else mask
    outer_mask = binary_dilation(mask, iterations=outer)
    ring = outer_mask & ~inner_mask
    report: Dict[str, object] = {
        "pipeline": "local_context_harmonization_v1",
        "ring_pixels": int(ring.sum()),
        "inner_ring_pixels": inner,
        "outer_ring_pixels": outer,
        "applied": False,
        "reason": None,
    }
    if int(ring.sum()) < 256:
        report["reason"] = "too_few_context_pixels"
        return repaired.copy(), report

    sigma_low = max(float(lowpass_sigma), 1.0)
    original_low = gaussian_filter(original, sigma=(sigma_low, sigma_low, 0.0))
    repaired_low = gaussian_filter(repaired, sigma=(sigma_low, sigma_low, 0.0))
    x = repaired_low[ring].astype(np.float64)
    y = original_low[ring].astype(np.float64)

    # A constant annulus cannot identify gain independently from offset.  Do
    # not turn a white synthetic fill black merely because its outside frame
    # is constant; leave such cases untouched.
    spread_x = np.percentile(x, 90, axis=0) - np.percentile(x, 10, axis=0)
    spread_y = np.percentile(y, 90, axis=0) - np.percentile(y, 10, axis=0)
    if float(np.max(spread_x)) < 0.02 or float(np.max(spread_y)) < 0.02:
        report["reason"] = "insufficient_context_variation"
        return repaired.copy(), report

    gains = np.ones(3, dtype=np.float64)
    offsets = np.zeros(3, dtype=np.float64)
    for channel in range(3):
        xc, yc = x[:, channel], y[:, channel]
        design = np.stack([xc, np.ones_like(xc)], axis=1)
        weights = np.ones_like(xc)
        beta = np.array([1.0, 0.0])
        # Huber IRLS rejects flowers/pattern elements that an image model moved
        # by a few pixels; low-frequency tone remains a stable paired signal.
        for _ in range(6):
            root_w = np.sqrt(weights)[:, None]
            beta, *_ = np.linalg.lstsq(design * root_w, yc * root_w[:, 0], rcond=None)
            residual = yc - design @ beta
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1.0e-6
            limit = 1.5 * scale
            weights = np.minimum(1.0, limit / np.maximum(np.abs(residual), 1.0e-12))
        gains[channel] = np.clip(beta[0], 0.67, 1.50)
        offsets[channel] = np.clip(beta[1], -0.25, 0.25)

    colour_matched = np.clip(repaired * gains[None, None, :] + offsets[None, None, :], 0.0, 1.0)

    downstream = max(float(downstream_blur_sigma), 0.0)
    if not np.isfinite(downstream):
        downstream = 0.0

    def rendered_high_frequency_rms(image: np.ndarray, applied: float) -> float:
        """Sharpness after this blur *and* the caller's downstream low-pass."""
        total = float(np.hypot(float(applied), downstream))
        if total <= 0.0:
            return high_frequency_rms(image)
        return high_frequency_rms(
            gaussian_filter(image, sigma=(total, total, 0.0)))

    def high_frequency_rms(image: np.ndarray) -> float:
        low = gaussian_filter(image, sigma=(1.0, 1.0, 0.0))
        values = (image - low)[ring]
        return float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))

    original_hf = high_frequency_rms(original)
    repaired_hf_before = high_frequency_rms(colour_matched)
    rendered_hf_before = rendered_high_frequency_rms(colour_matched, 0.0)
    blur_sigma = 0.0
    harmonized = colour_matched
    # Only remove excess sharpness.  We never sharpen a soft generation,
    # because sharpening invented texture tends to ring at the patch boundary.
    # The comparison is made on the *rendered* result, so a repair the splat
    # kernel will soften on its own is left alone here.
    if original_hf > 1.0e-5 and rendered_hf_before > original_hf * 1.08:
        target = original_hf
        candidates = np.linspace(0.25, max(float(maximum_blur_sigma), 0.25), 12)
        best_error = abs(rendered_hf_before - target)
        for candidate in candidates:
            error = abs(
                rendered_high_frequency_rms(colour_matched, float(candidate)) - target)
            if error < best_error:
                harmonized = gaussian_filter(
                    colour_matched, sigma=(float(candidate), float(candidate), 0.0))
                blur_sigma, best_error = float(candidate), error

    # A single RGB affine cannot follow spatially varying illumination or an
    # image model that changed the rug's tone differently on each side.  Build
    # a very-low-frequency correction field from the safe annulus and extend
    # it inward by nearest boundary value followed by a wide blur.  This uses
    # no original pixel from inside the invalid hole and cannot copy its old
    # contents back into the repair.
    current_low = gaussian_filter(harmonized, sigma=(sigma_low, sigma_low, 0.0))
    annulus_delta = original_low - current_low
    _distance, nearest = distance_transform_edt(~ring, return_indices=True)
    spatial_delta = annulus_delta[nearest[0], nearest[1]]
    correction_sigma = max(6.0, 0.35 * float(outer))
    spatial_delta = gaussian_filter(
        spatial_delta, sigma=(correction_sigma, correction_sigma, 0.0)
    )
    spatial_delta = np.clip(spatial_delta, -0.18, 0.18)
    harmonized = np.clip(harmonized + spatial_delta, 0.0, 1.0)

    repaired_hf_after = high_frequency_rms(harmonized)
    before_mae = float(np.mean(np.abs(repaired_low[ring] - original_low[ring])))
    harmonized_low = gaussian_filter(harmonized, sigma=(sigma_low, sigma_low, 0.0))
    after_mae = float(np.mean(np.abs(harmonized_low[ring] - original_low[ring])))
    report.update({
        "applied": True,
        "reason": "ok",
        "rgb_gain": [float(value) for value in gains],
        "rgb_offset": [float(value) for value in offsets],
        "blur_sigma_pixels": float(blur_sigma),
        "downstream_blur_sigma_pixels": float(downstream),
        "sharpness_target_semantics": (
            "rendered_after_downstream_kernel" if downstream > 0.0
            else "image_only"
        ),
        "spatial_colour_correction_sigma_pixels": float(correction_sigma),
        "spatial_colour_correction_abs_p90": float(
            np.percentile(np.abs(spatial_delta[mask]), 90.0)
        ),
        "context_low_frequency_mae_before": before_mae,
        "context_low_frequency_mae_after": after_mae,
        "context_high_frequency_rms_original": original_hf,
        "context_high_frequency_rms_before": repaired_hf_before,
        "context_high_frequency_rms_after": repaired_hf_after,
    })
    return np.clip(harmonized, 0.0, 1.0).astype(np.float32), report


def _write_float_png(path: Path, image: np.ndarray) -> Dict[str, object]:
    from PIL import Image

    array = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    payload = buffer.getvalue()
    with path.open("xb") as handle:
        handle.write(payload)
    return {
        "path": str(path),
        "bytes": int(len(payload)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": [int(array.shape[1]), int(array.shape[0])],
    }


def _write_audit_manifest(path: Path, manifest: Dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _begin_repair_audit(
    audit_dir: str | Path | None,
    images: Sequence[np.ndarray],
    roles: Sequence[str],
    hole_mask: np.ndarray,
    config: RepairConfig,
    request_stats: Dict[str, object],
) -> Tuple[Path | None, Dict[str, object] | None]:
    if audit_dir is None:
        return None, None
    root = Path(audit_dir)
    root.mkdir(parents=True, exist_ok=True)
    token = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    attempt = root / f"api_repair_{token}"
    attempt.mkdir(exist_ok=False)
    inputs = []
    for index, (image, role) in enumerate(zip(images, roles), start=1):
        record = _write_float_png(attempt / f"input_{index}_{role}.png", image)
        record["index"] = int(index)
        record["role"] = str(role)
        inputs.append(record)
    mask_record = _write_float_png(
        attempt / "edit_mask.png", make_edit_mask_guide(hole_mask)
    )
    manifest: Dict[str, object] = {
        "schema_version": 1,
        "status": "requesting",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": asdict(config),
        "request": dict(request_stats),
        "inputs": inputs,
        "edit_mask": mask_record,
        "candidates": [],
    }
    _write_audit_manifest(attempt / "request.json", manifest)
    return attempt, manifest


def _audit_failure(
    attempt: Path | None,
    manifest: Dict[str, object] | None,
    *,
    status: str,
    error: Exception,
) -> None:
    if attempt is None or manifest is None:
        return
    manifest["status"] = str(status)
    manifest["error"] = {
        "type": type(error).__name__,
        "message": str(error)[:1000],
    }
    _write_audit_manifest(attempt / "request.json", manifest)


def _view_consistency_score(report: Dict[str, object]) -> float:
    """Rank only candidates that already passed ``align_snapshot_repair``."""

    correlation = float(report.get("context_correlation", 0.0))
    mae = float(report.get("context_low_frequency_mae", 1.0))
    reprojection = float(report.get("p90_reprojection_error_pixels", 100.0))
    rotation = abs(float(report.get("rotation_degrees", 180.0)))
    return correlation - 2.0 * mae - 0.01 * reprojection - 0.01 * rotation


def repair_snapshot(
    snapshot_rgb: np.ndarray,
    hole_mask: np.ndarray,
    *,
    reference_rgbs: Sequence[np.ndarray] = (),
    blank_hole: bool = False,
    api_key: str | None = None,
    config: RepairConfig | None = None,
    feather_pixels: int = 3,
    session=None,
    return_raw: bool = False,
    audit_dir: str | Path | None = None,
) -> (
    Tuple[np.ndarray, Dict[str, object]]
    | Tuple[np.ndarray, np.ndarray, Dict[str, object]]
):
    """整条链路：上传完整侧视参考和破损主视图 -> 取回结果 -> 本地按掩膜合回。

    图1固定为一张完整侧视参考，图2固定为唯一待修的破损主视图；掩膜不作为
    API 输入，也不会先涂灰。它只在模型返回之后用于本地纹理回贴。

    默认返回 ``(composited, stats)``，与旧调用完全兼容。显式传入
    ``return_raw=True`` 时返回 ``(composited, raw_produced, stats)``；其中
    ``raw_produced`` 保持模型输出的原始分辨率，供下游先做配准与外观协调。
    """

    repair_started = time.monotonic()
    config = config or RepairConfig()
    config.validate()
    images, roles, effective_config, request_stats = prepare_mask_guided_request(
        snapshot_rgb,
        hole_mask,
        reference_rgbs,
        config,
        blank_hole=bool(blank_hole),
    )
    prepared_at = time.monotonic()
    refs = [encode_data_uri(image) for image in images]
    encoded_at = time.monotonic()
    request_stats["input_data_uri_sha256"] = [
        hashlib.sha256(value.encode("ascii")).hexdigest() for value in refs
    ]
    attempt, audit_manifest = _begin_repair_audit(
        audit_dir,
        images,
        roles,
        hole_mask,
        effective_config,
        request_stats,
    )
    request_started = time.monotonic()
    try:
        urls, body = request_repair(
            refs, api_key=api_key, config=effective_config, session=session
        )
    except Exception as exc:
        _audit_failure(attempt, audit_manifest, status="request_failed", error=exc)
        raise
    request_finished = time.monotonic()

    candidate_records: List[Dict[str, object]] = []
    accepted = []
    download_seconds = 0.0
    for index, url in enumerate(urls, start=1):
        record: Dict[str, object] = {"index": int(index), "accepted": False}
        try:
            download_started = time.monotonic()
            candidate = fetch_image(url, session=session)
            candidate_download_seconds = time.monotonic() - download_started
            download_seconds += candidate_download_seconds
            record["download_seconds"] = float(candidate_download_seconds)
            record["size"] = [int(candidate.shape[1]), int(candidate.shape[0])]
            if attempt is not None:
                record["artifact"] = _write_float_png(
                    attempt / f"candidate_{index:02d}_raw.png", candidate
                )
            if bool(config.enforce_view_consistency):
                (
                    _aligned_original,
                    aligned_candidate,
                    _aligned_mask,
                    alignment,
                ) = align_snapshot_repair(
                    snapshot_rgb,
                    candidate,
                    hole_mask,
                    maximum_rotation_degrees=float(
                        config.maximum_view_rotation_degrees
                    ),
                )
                score = _view_consistency_score(alignment)
                record["alignment"] = alignment
                record["score"] = float(score)
                record["accepted"] = True
                accepted.append((float(score), index, candidate, aligned_candidate))
            else:
                record["score"] = float(-index)
                record["accepted"] = True
                accepted.append((float(-index), index, candidate, candidate))
        except Exception as exc:
            record["rejection"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
            }
        candidate_records.append(record)

    if audit_manifest is not None:
        audit_manifest["request_id"] = (
            body.get("request_id") if isinstance(body, dict) else None
        )
        audit_manifest["candidates"] = candidate_records
    if not accepted:
        reasons = "; ".join(
            str(record.get("rejection", {}).get("message", "未知原因"))
            for record in candidate_records
        )
        exc = RuntimeError(
            "AI 修复返回的候选均未通过主视角一致性检查，已拒绝采用："
            + reasons
            + (f"；原始候选审计：{attempt}" if attempt is not None else "")
        )
        _audit_failure(attempt, audit_manifest, status="candidates_rejected", error=exc)
        raise exc

    _score, selected_index, produced, preview_source = max(
        accepted, key=lambda item: item[0]
    )
    composited, stats = composite_repair(
        snapshot_rgb, preview_source, hole_mask, feather_pixels=feather_pixels
    )
    timing_seconds = {
        "prepare_inputs": float(prepared_at - repair_started),
        "encode_inputs": float(encoded_at - prepared_at),
        "audit_before_request": float(request_started - encoded_at),
        "api_request_until_url": float(request_finished - request_started),
        "download_decode": float(download_seconds),
        "total": float(time.monotonic() - repair_started),
    }
    stats.update(
        {
            "model": config.model,
            "input_images": len(refs),
            "input_layout": roles,
            "reference_images": request_stats["material_references_used"],
            "reference_images_dropped": request_stats[
                "material_references_dropped"
            ],
            "hole_blanked_before_sending": False,
            "candidates_returned": len(urls),
            "candidate_reports": candidate_records,
            "selected_candidate": int(selected_index),
            "view_consistency_enforced": bool(config.enforce_view_consistency),
            "produced_size": [int(produced.shape[1]), int(produced.shape[0])],
            "request_id": body.get("request_id") if isinstance(body, dict) else None,
            "config": asdict(config),
            "effective_prompt": effective_config.prompt,
            "audit_path": str(attempt) if attempt is not None else None,
            "timing_seconds": timing_seconds,
        }
    )
    if attempt is not None and audit_manifest is not None:
        audit_manifest["status"] = "accepted"
        audit_manifest["timing_seconds"] = timing_seconds
        audit_manifest["selected_candidate"] = int(selected_index)
        audit_manifest["preview"] = _write_float_png(
            attempt / "masked_preview.png", composited
        )
        audit_manifest["result_stats"] = stats
        _write_audit_manifest(attempt / "request.json", audit_manifest)
    if bool(return_raw):
        return composited, produced, stats
    return composited, stats
