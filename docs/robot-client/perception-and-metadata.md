# Perception And Metadata

The client-side perception package does not run object detection or scene graph inference. It collects robot-local observations and sends them to the server as metadata next to captured image bytes.

Server-side interpretation of this payload is documented in [`../server/detection-tracking-and-fusion.md`](../server/detection-tracking-and-fusion.md) and [`../server/data-models-and-contracts.md`](../server/data-models-and-contracts.md).

## Main Data Flow

The shared capture path is `TurnManager._capture_with_metadata`:

1. `camera_adapter.capture_frame(frame_id)` returns JPEG bytes, dimensions, timestamp, and camera FOV.
2. `robot_context.snapshot()` returns pose, people, social people, and sonar.
3. `metadata_builder.build(...)` combines capture and context into one server-compatible dict.
4. `PepperServerTransport` JSON-serializes that dict into multipart field `metadata`.

This path is used for quick look, visual refresh, panorama scans, and sequential scans.

## CameraAdapter

File: `client/app/scripts/pepper_client/perception/camera_adapter.py`

`CameraAdapter` wraps `ALVideoDevice`.

### `capture_frame(frame_id=None)`

Steps:

1. Reads capture config.
2. Builds a unique subscriber name from app id and current time.
3. Calls `ALVideoDevice.subscribeCamera` with camera id, resolution, color space, and FPS.
4. Calls `getImageRemote`.
5. Extracts width, height, and raw bytes.
6. Builds a PIL RGB image.
7. Encodes JPEG with configured quality.
8. Reads horizontal and vertical FOV if available.
9. Returns a dict with image bytes and camera metadata.
10. Always tries to unsubscribe the camera handle in `finally`.

Returned dict shape:

```json
{
  "frame_id": "pepper_frame_...",
  "image_bytes": "<bytes>",
  "image_width": 640,
  "image_height": 480,
  "timestamp": 1770000000.0,
  "camera_hfov": 0.998,
  "camera_vfov": 0.773
}
```

Failure behavior:

- Missing `ALVideoDevice` raises `CameraCaptureError`.
- Bad/no image raises `CameraCaptureError`.
- Any other capture exception is wrapped as `CameraCaptureError`.

### Raw Image Conversion

`_build_image` uses `Image.frombytes("RGB", (width, height), raw_data)`. It falls back to `Image.fromstring` for older Pillow/Python compatibility.

`_coerce_bytes` handles bytearray, bytes, and NAOqi buffer-like data.

### FOV Reading

`_camera_fov` tries:

- `ALVideoDevice.getHorizontalFOV(camera_id)`.
- `ALVideoDevice.getVerticalFOV(camera_id)`.

If either fails, that value is `None`. The server can still process image data without FOV, but geometry fusion is weaker.

## FakeCameraAdapter

File: `client/app/scripts/pepper_client/perception/camera_adapter.py`

`FakeCameraAdapter` is selected when:

- `capture.fake_camera=true`.
- `capture.fake_camera_path` is non-empty.

It reads all `.png`, `.jpg`, and `.jpeg` files from the folder and randomly picks one on each capture. It returns JPEG bytes and fake FOV values `0.5, 0.5`.

This is useful for local virtual robot tests. It is not a good geometry simulator because FOV and pose values are artificial.

## PoseAdapter

File: `client/app/scripts/pepper_client/perception/pose_adapter.py`

`PoseAdapter` wraps `ALMotion`.

Methods:

| Method | Behavior |
|---|---|
| `snapshot()` | Returns `head_yaw`, `head_pitch`, and `body_yaw`. |
| `get_head_angles()` | Reads `HeadYaw` and `HeadPitch` through `getAngles`. |
| `get_body_yaw()` | Reads robot position and extracts yaw from index 2. |
| `move_head(yaw, pitch, speed)` | Calls `ALMotion.setAngles`. |
| `restore_head(pose, speed)` | Moves head back to stored yaw/pitch and waits briefly. |

If `ALMotion` is missing, pose defaults are safe: head yaw/pitch `0.0`, body yaw `None`, and moves are skipped.

## Scan Planner

File: `client/app/scripts/pepper_client/perception/scan_planner.py`

This module is deliberately small and pure. It reads config and returns scan decisions.

Functions:

| Function | Output |
|---|---|
| `planned_yaws_radians(config)` | Converts `capture.scan_yaws_deg` to radians. |
| `scan_pitch(config)` | Returns `capture.scan_head_pitch`. |
| `scan_mode(config)` | Returns `panorama_detect` or `sequential_detect`. |
| `stick_together(config)` | Returns `panorama.stick_together`. |
| `summary_after_scan(config)` | Returns `panorama.summary_after_scan` or legacy behavior fallback. |
| `memory_render_limit(config)` | Returns tablet memory render limit. |

If panorama is disabled, `scan_mode` returns `sequential_detect`.

## PeopleAdapter

File: `client/app/scripts/pepper_client/perception/people_adapter.py`

`PeopleAdapter` wraps `ALPeoplePerception` and `ALMemory`.

### Start/Stop

If `social.enable_people_perception=true`, `start()` subscribes to `ALPeoplePerception` with a subscription name derived from service name.

`stop()` unsubscribes if subscribed.

### `snapshot_people()`

Reads visible people from ALMemory:

- `PeoplePerception/PeopleList`.
- `PeoplePerception/Person/<id>/IsVisible`.
- `PeoplePerception/Person/<id>/AnglesYawPitch`.
- `PeoplePerception/Person/<id>/Distance`.

Returned person shape:

```json
{
  "id": 37195,
  "yaw": 0.132,
  "pitch": -0.006,
  "distance": 0.746
}
```

These records become `metadata.people` and are used server-side for person fusion and angular matching.

## FaceAdapter

File: `client/app/scripts/pepper_client/perception/face_adapter.py`

`FaceAdapter` wraps `ALFaceDetection` and reads `FaceDetected` from ALMemory.

### `snapshot_faces()`

It parses the `FaceDetected` payload and extracts:

- Face yaw.
- Face pitch.
- Recognized face label, when present.
- Confidence, chosen as the best value between 0 and 1 in the face metadata block.

Returned face shape:

```json
{
  "yaw": 0.1,
  "pitch": -0.02,
  "face_label": "Alice",
  "face_confidence": 0.83
}
```

### `match_faces_to_people(people)`

Matches face records to people records by angular proximity.

Matching score:

```text
abs(person.yaw - face.yaw) + abs(person.pitch - face.pitch)
```

A match is accepted only if the delta is below `social.face_match_max_angle_rad`. If multiple faces match the same person id, the highest confidence face wins.

The returned dict maps person id to face metadata.

## SocialAdapter

File: `client/app/scripts/pepper_client/perception/social_adapter.py`

`SocialAdapter` enriches people records with social attributes from Pepper services. It does not find people by itself. It receives the visible people list from `PeopleAdapter`.

### Subscribed Services

Depending on config, it subscribes to:

- `ALFaceCharacteristics`.
- `ALGazeAnalysis`.
- `ALEngagementZones`.
- `ALSittingPeopleDetection`.
- `ALWavingDetection`.

It also uses `FaceAdapter.match_faces_to_people` to add face labels/confidence.

### `snapshot_social_people(people)`

For every visible person, it reads social keys and returns a list of enriched payloads. A payload is included if it has more than just `id` and `timestamp`.

Possible fields:

| Field | Source / meaning |
|---|---|
| `id` | Pepper person id. |
| `timestamp` | Local client timestamp. |
| `age` | From age properties. |
| `age_confidence` | Confidence for age. |
| `age_bucket` | Derived child/teen/adult/senior bucket. |
| `gender_code` | Numeric gender code. |
| `gender_confidence` | Confidence for gender. |
| `gender` | Derived `female`, `male`, or `unknown`. |
| `smile_score` | Smile score. |
| `smile_confidence` | Smile confidence. |
| `expression` | Best expression label. |
| `expression_confidence` | Best expression score. |
| `expression_scores` | Raw expression score list. |
| `is_looking_at_robot` | Boolean ALMemory flag. |
| `looking_at_robot_score` | Numeric looking-at-robot score. |
| `head_angles` | Head angle list. |
| `gaze_direction` | Two-item list for left/right and up/down direction. |
| `engagement_zone` | Numeric engagement zone. |
| `is_sitting` | Boolean sitting flag. |
| `is_waving` | Combined waving flag. |
| `is_waving_center` | Center waving flag. |
| `is_waving_left` | Left waving flag. |
| `is_waving_right` | Right waving flag. |
| `eyes_opened` | Eye opening degree list. |
| `face_label` | Face recognition label from `FaceAdapter`. |
| `face_confidence` | Face recognition confidence. |

Current implementation detail: `gaze_direction` is a two-item list, not a string. Some older logs may show a string from previous versions.

## SonarAdapter

File: `client/app/scripts/pepper_client/perception/sonar_adapter.py`

`SonarAdapter` subscribes to `ALSonar` if enabled and reads:

- `Device/SubDeviceList/US/Left/Sensor/Value`.
- `Device/SubDeviceList/US/Right/Sensor/Value`.

Returned shape:

```json
{
  "left": 0.7,
  "right": 0.8
}
```

If both values are missing, it returns `None` and metadata omits sonar.

## RobotContextCollector

File: `client/app/scripts/pepper_client/perception/robot_context.py`

This class coordinates pose, people, social, and sonar adapters.

Lifecycle:

- `start()` starts people, social, and sonar collectors.
- `stop()` stops them.

Snapshot shape:

```json
{
  "pose": {"head_yaw": 0.0, "head_pitch": 0.0, "body_yaw": null},
  "people": [{"id": 37195, "yaw": 0.13, "pitch": -0.01, "distance": 0.75}],
  "social_people": [{"id": 37195, "is_sitting": true, "gender": "female"}],
  "sonar": {"left": 0.7, "right": 0.8}
}
```

## MetadataBuilder

File: `client/app/scripts/pepper_client/utils/metadata_builder.py`

`MetadataBuilder.build(capture, context, frame_id, scan_id=None, capture_mode=None)` creates the JSON payload sent to the server.

Output fields:

| Field | Source |
|---|---|
| `head_yaw` | `context.pose.head_yaw`, default `0.0`. |
| `head_pitch` | `context.pose.head_pitch`, default `0.0`. |
| `body_yaw` | `context.pose.body_yaw`. |
| `camera_hfov` | `capture.camera_hfov`. |
| `camera_vfov` | `capture.camera_vfov`. |
| `image_width` | `capture.image_width`. |
| `image_height` | `capture.image_height`. |
| `timestamp` | `capture.timestamp`. |
| `frame_id` | Generated client frame id. |
| `scan_id` | Generated client scan id for scan frames. |
| `people` | `context.people` list. |
| `social_people` | Included only when non-empty. |
| `sonar` | Included only when present. |
| `capture_mode` | `caption`, `detect`, or `scan` when provided. |

This output is expected to match the server `RobotMetadata` schema.

## Geometry And Server Matching

The most important client-side fields for server-side geometry are:

- `head_yaw`.
- `head_pitch`.
- `body_yaw`.
- `camera_hfov`.
- `camera_vfov`.
- `image_width`.
- `image_height`.
- `people[].yaw`.
- `people[].pitch`.
- `people[].distance`.
- `people[].id`.

The server can map detected bounding boxes into approximate camera angles using image dimensions and FOV, then match those angles against Pepper people perception yaw/pitch. The social metadata can then be attached to the matched tracked person.

If any FOV or image size values are missing or fake, that geometry match becomes less reliable.

## Where To Add New Robot Metadata

To add a new metadata field from Pepper:

1. Read the ALMemory key or robot service in the relevant adapter.
2. Add it to `people`, `social_people`, `sonar`, or a new context section.
3. Add it to `MetadataBuilder` if it is top-level or optional.
4. Add the matching Pydantic field on the server side in `server/app/schemas/robot.py`.
5. Add server usage in memory fusion or scene graph enhancement.
6. Update this document and server docs.
