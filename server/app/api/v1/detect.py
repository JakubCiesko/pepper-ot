import asyncio
import logging
from typing import Annotated

import app.api.v1.image_utils as img_utils
from app.core.runtime.state import app_state
from app.orchestration.services.detection import DetectService
from app.schemas.detect import DetectFormRequest
from app.schemas.detect import DetectionResponse
from app.schemas.robot import RobotMetadata
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/detect", response_model=DetectionResponse)
async def detect_endpoint(
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
    publish: bool = Form(True),
    resize_image: bool = Form(True),
):
    """
    Run the perception pipeline on an uploaded image and return detected objects.

    Behavior:
    - Reads the uploaded image bytes.
    - Parses optional robot metadata from JSON form field.
    - Executes detection via DetectService (local pipeline or worker-backed runtime).
    - Optionally publishes resulting state/events to websocket subscribers.

    Args:
        file: Uploaded image file (multipart/form-data).
        metadata: Optional JSON string with robot metadata (pose, camera info, frame IDs).
        publish: If True, allows service-layer broadcasting/persistence side effects.

    Returns:
        DetectionResponse including:
            - request/response id
            - detected objects
            - timestamp
            - image dimensions

    Raises:
        HTTPException: Propagated from DetectService for invalid image/metadata
        or runtime processing errors.
    """
    form = DetectFormRequest(
        metadata=metadata,
        publish=publish,
        resize_image=resize_image,
    )

    # TODO: instantiating service each time?
    logger.info(
        "Detection endpoint called with file=%s, metadata=%s, publish=%s",
        file.filename,
        form.metadata,
        form.publish,
    )
    image_bytes = await file.read()
    image_bytes, (w, h) = (
        img_utils.resize_image_bytes(image_bytes, debug_show=True)
        if form.resize_image
        else (image_bytes, (None, None))
    )
    service = DetectService(app_state)
    robot_metadata = service.parse_metadata(form.metadata)
    if w is not None and h is not None:
        robot_metadata.image_width, robot_metadata.image_height = w, h
    logger.info("Running detection with received robot metadata: %s", robot_metadata)
    response = await service.process(image_bytes, robot_metadata, form.publish)
    logger.info("Detection endpoint completed: %s", response.id)
    return response


@router.post("/detect/panorama", response_model=DetectionResponse)
async def panorama_detect_endpoint(
    files: Annotated[list[UploadFile], File(...)],
    metadata: Annotated[list[str | None], Form(...)],
    publish: bool = Form(True),
    resize_image: bool = Form(True),
    stick_together: bool = Form(True),
):
    """
    Detect objects from multiple images with corresponding metadata.

    Modes:
    - stick_together=True:
        Stitches images into a panorama and runs a single detection.
    - stick_together=False:
        Processes each image independently in parallel and merges results.

    Each image must have its own metadata entry, matched by index.
    """
    if len(files) != len(metadata):
        raise HTTPException(
            status_code=400,
            detail="The number of files must match the number of metadata entries. Pass empty string for no metadata.",
        )
    logger.info(
        "Panorama detection called with %d files, stick_together=%s",
        len(files),
        stick_together,
    )
    service = DetectService(app_state)
    image_bytes_list: list[bytes] = []
    robot_metadata_list = []

    for file, meta in zip(files, metadata, strict=True):
        image_bytes = await file.read()
        image_bytes, (w, h) = (
            img_utils.resize_image_bytes(image_bytes, debug_show=True)
            if resize_image
            else (image_bytes, (None, None))
        )
        image_bytes_list.append(image_bytes)
        data = service.parse_metadata(meta)
        if w is not None and h is not None:
            data.image_width, data.image_height = w, h
        robot_metadata_list.append(data)

    if stick_together:
        logger.info("Creating panorama from %d images", len(image_bytes_list))
        panorama_bytes = img_utils.create_panorama(image_bytes_list)
        robot_metadata = RobotMetadata.merge_robot_metadata_for_panorama(
            robot_metadata_list
        )
        logger.info("Merged robot metadata into one metadata: %s", robot_metadata)
        response = await service.process(
            panorama_bytes,
            robot_metadata,
            publish,
        )
        return response

    async def process_single(
        image_bytes: bytes,
        robot_metadata: RobotMetadata,
        index: int,
    ):
        image_bytes, (w, h) = (
            img_utils.resize_image_bytes(image_bytes, debug_show=True)
            if resize_image
            else (image_bytes, (None, None))
        )
        if w is not None and h is not None:
            robot_metadata.image_width, robot_metadata.image_height = w, h

        def run_detection():
            local_service = DetectService(app_state)
            return asyncio.run(
                local_service.process(
                    image_bytes,
                    robot_metadata,
                    publish,
                )
            )

        result = await asyncio.to_thread(run_detection)

        logger.info(
            "Processed image %d/%d -> %s",
            index + 1,
            len(image_bytes_list),
            result.id,
        )
        return result

    tasks = [
        process_single(img_bytes, meta, idx)
        for idx, (img_bytes, meta) in enumerate(
            zip(image_bytes_list, robot_metadata_list, strict=True)
        )
    ]

    results = await asyncio.gather(*tasks)

    merged_objects = []
    for res in results:
        merged_objects.extend(res.objects)
    base_response = results[0]

    merged_response = DetectionResponse(
        id=base_response.id,
        objects=merged_objects,
        timestamp=base_response.timestamp,
        image_width=base_response.image_width,
        image_height=base_response.image_height,
    )

    logger.info(
        "Merged %d detections from %d images",
        len(merged_objects),
        len(results),
    )
    # only merge together later?
    return merged_response
