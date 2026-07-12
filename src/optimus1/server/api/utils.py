import base64
import binascii
import logging
import os
import re
from typing import Any, Dict, List

try:
    import shortuuid
except ModuleNotFoundError:  # minimal fallback for lean server images
    import uuid

    class _ShortUUID:
        @staticmethod
        def uuid():
            return uuid.uuid4().hex

    shortuuid = _ShortUUID()

logger = logging.getLogger(__name__)
_MAX_IMAGE_BYTES = int(os.getenv("CACT_MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))


def _safe_component(value: str, default: str = "item") -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))[:128]
    return value or default


def _decode_image(value: str) -> bytes:
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 image") from exc
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError("image exceeds CACT_MAX_IMAGE_BYTES")
    return data


def base64lst2img_path(base64_lst: List[str] | None, image_root: str):
    """
    Convert a list of base64-encoded images to actual image files and save them.

    Args:
        base64_lst (List[str]): A list of base64-encoded images.

    Returns:
        List[str]: A list of image file names that were saved.

    """
    # image_root = "api/imgs"
    os.makedirs(image_root, exist_ok=True)
    image_file_names = []
    if base64_lst is None:
        return []

    for idx, image_byte in enumerate(base64_lst):
        uuid = shortuuid.uuid()
        imgdata = _decode_image(image_byte)
        image_file = os.path.join(image_root, f"{uuid}_{idx}.jpg")

        with open(image_file, "wb") as f:
            f.write(imgdata)
        # logger.info(f"Save image to {image_file}")

        image_file_names.append(image_file)
    return image_file_names


def base64_to_image(
    rgb_images: List[Dict[str, Any]],
    image_root: str = "api/imgs",
    task: str = "plan|action|reflection|replan",
    step: int = 0,
) -> List[str]:
    """
    Convert a list of base64-encoded images to actual image files and save them.

    Args:
        rgb_images (List[Dict[str, Any]]): A list of dictionaries containing base64-encoded images.
        image_root (str, optional): The root directory where the image files will be saved. Defaults to "api/imgs".
        task (str, optional): The task name used in the image file name. Defaults to "plan|action|reflection|replan".
        step (int, optional): The step number used in the image file name. Defaults to 0.

    Returns:
        List[str]: A list of image file names that were saved.

    """
    os.makedirs(image_root, exist_ok=True)
    task = _safe_component(task, "task")
    image = rgb_images[-1]
    uuid = shortuuid.uuid()[:5]

    image_byte = image["image"]
    imgdata = _decode_image(image_byte)
    image_file = os.path.join(image_root, f"{task}_{uuid}_{step}.jpg")

    with open(image_file, "wb") as f:
        f.write(imgdata)
    return [image_file]


def base64_to_image2(
    rgb_images: List[Dict[str, Any]], image_root: str = "api/imgs"
) -> List[str]:
    if not rgb_images:
        print("none images")
        return []  # 如果输入列表为空，则直接返回空列表

    uuid = shortuuid.uuid()
    last_image_file = ""  # 初始化变量来存储最后一张图像的文件名

    # 仅处理最后一张图片

    image = rgb_images[-1]  # 获取最后一张图像的数据
    image_byte = image["image"]
    imgdata = _decode_image(image_byte)

    # 构建文件名
    if "yaw" in image and "pitch" in image:
        yaw = image["yaw"]
        pitch = image["pitch"]
        last_image_file = os.path.join(image_root, f"{uuid}_{yaw}_{pitch}.jpg")
    else:
        last_image_file = os.path.join(image_root, f"{uuid}.jpg")

    # 打印并保存图像
    print("Save image to ", last_image_file)
    with open(last_image_file, "wb") as f:
        f.write(imgdata)

    # 由于函数定义返回一个列表，这里我们返回包含最后一张图像文件名的列表
    # 如果你只需要返回一个字符串，这里可以直接返回 last_image_file
    return [last_image_file]
