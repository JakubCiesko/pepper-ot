# Client Configuration

The robot client configuration lives in `client/app/scripts/client_config.json`. Defaults and normalization are implemented in `client/app/scripts/pepper_client/utils/config.py`.

The config file is packaged with the robot application through `client/app/pepper-grounded-client.pml`, so changes to the deployed default config need to be included before packaging or copied to the robot app directory.

## Loading Rules

`load_config(path, logger=None)`:

1. Starts with a deep copy of `DEFAULT_CONFIG`.
2. If `client_config.json` exists, parses JSON and deep-merges it into defaults.
3. Calls `normalize_config`.
4. Stores `_config_path` in the runtime dict.
5. Returns the runtime config dict.

This means missing keys are safe as long as defaults exist.

## Saving Rules

`save_config(config, logger=None)`:

1. Reads `_config_path`.
2. Copies config.
3. Removes `_config_path`.
4. Writes formatted JSON with sorted keys.

The service methods `setDialogLanguage`, `setServerBaseUrl`, and `reloadConfig` use this machinery.

## Runtime Reload

`PepperGroundedClient.reloadConfig()` reloads JSON and calls `_apply_loaded_config`.

Hot reload behavior is partial:

- The transport receives updated config immediately.
- The dialog adapter receives updated config immediately.
- The camera adapter is recreated.
- The tablet adapter is recreated.
- The turn manager is kept but receives the new camera/tablet references.
- Face/context collectors are stopped and restarted.

If you add a new config field that is read only during adapter construction, ensure that adapter is recreated or has `update_config` called during reload.

## `app` Section

```json
"app": {
  "app_id": "PepperGroundedClient",
  "service_name": "PepperGroundedClient"
}
```

`app_id` is used for camera subscription names and logging identity. `service_name` is used for robot service subscription names and is expected to match the service registered in `manifest.xml`.

Changing these names affects QiChat calls, tablet JS service lookup, and app packaging. Do not rename them unless you update all references.

## `server` Section

This section controls server base URL, endpoint paths, timeouts, TLS verification, and whether server-side events are published.

Important keys:

| Key | Used by | Meaning |
|---|---|---|
| `base_url` | `PepperServerTransport` | Server origin, without trailing slash after normalization. |
| `caption_path` | `transport.caption` | Caption endpoint path. |
| `detect_path` | `transport.detect` | Single-frame detect endpoint path. |
| `detect_panorama_path` | `transport.panorama_detect` | Multi-frame panorama detect endpoint path. |
| `chat_path` | `transport.chat` | General/object chat endpoint path. |
| `pregenerate_qa_path` | `transport.pregenerate_qa` | Q/A pool read/generation endpoint path. |
| `memory_summary_path` | `transport.memory_summary` | Memory summary endpoint path. |
| `memory_reset_path` | `transport.reset_memory` | Memory reset endpoint path. |
| `model_facing_language` | `transport.chat` | Optional server field for model-facing language. Usually `null`. |
| `publish` | detect/caption calls | Whether server should publish dashboard/WebSocket events. |
| `verify_tls` | `requests` | Whether HTTPS certificates are verified. |
| `*_timeout_seconds` | all transport calls | Request timeout per endpoint class. |

Server endpoint contracts are documented in [`../server/api-reference.md`](../server/api-reference.md).

## `capture` Section

This section controls camera capture and scan movement.

Important keys:

| Key | Used by | Meaning |
|---|---|---|
| `camera_id` | `CameraAdapter.subscribeCamera` | Pepper camera id. |
| `resolution` | `CameraAdapter.subscribeCamera` | NAOqi resolution enum value. |
| `color_space` | `CameraAdapter.subscribeCamera` | NAOqi color space enum value. Current code expects RGB-compatible bytes. |
| `fps` | `CameraAdapter.subscribeCamera` | Camera subscription FPS. |
| `jpeg_quality` | `CameraAdapter._encode_jpeg` | JPEG quality for server upload. |
| `scan_yaws_deg` | `scan_planner.planned_yaws_radians` | Head yaw sequence for scan captures. |
| `scan_head_pitch` | `scan_planner.scan_pitch` | Head pitch for scan captures. |
| `head_move_speed` | `PoseAdapter.move_head` | Head movement speed fraction. |
| `settle_seconds` | `_prepare_scan_captures` | Wait after head movement before capture. |
| `refresh_ttl_seconds` | `_run_ask` | How old visual context can be before auto refresh. |
| `frame_prefix` | `ids.new_frame_id` | Prefix for generated frame IDs. |
| `scan_prefix` | `ids.new_scan_id` | Prefix for generated scan IDs. |
| `scan_summary_query_en` | `_scan_summary_query` | English scan summary prompt. |
| `scan_summary_query_cs` | `_scan_summary_query` | Czech scan summary prompt. |
| `fake_camera` | `_initialize_camera_adapter` | Enables local fake camera. |
| `fake_camera_path` | `FakeCameraAdapter` | Folder of images for fake captures. |

`normalize_config` ensures `scan_yaws_deg` is a non-empty list, but does not deeply validate camera enum values.

## `behavior` Section

This section controls high-level turn behavior.

| Key | Meaning |
|---|---|
| `caption_run_detect` | Quick look caption also asks server to run detect/memory update. |
| `caption_retry_on_timeout` | Caption endpoint retries once after timeout. |
| `auto_refresh_before_chat` | General ask refreshes visual context if stale. |
| `allow_scan_summary_chat` | Legacy fallback for scan summary when panorama section lacks `summary_after_scan`. |
| `speak_acknowledgements` | Turn manager speaks an acknowledgement before long-running work. |
| `auto_restore_head_pose` | Scan restores original head yaw/pitch after capture sequence. |
| `max_query_chars` | Query truncation length before sending chat/object/cached-answer requests. |

## `social` Section

This section controls subscriptions and metadata extraction from Pepper people/social services.

| Key | Used by | Meaning |
|---|---|---|
| `enable_people_perception` | `PeopleAdapter` | Subscribe to `ALPeoplePerception`. |
| `enable_face_detection` | `FaceAdapter` | Subscribe to `ALFaceDetection`. |
| `enable_face_characteristics` | `SocialAdapter` | Subscribe to `ALFaceCharacteristics`. |
| `enable_gaze_analysis` | `SocialAdapter` | Subscribe to `ALGazeAnalysis`. |
| `enable_engagement_zones` | `SocialAdapter` | Subscribe to `ALEngagementZones`. |
| `enable_sitting_detection` | `SocialAdapter` | Subscribe to `ALSittingPeopleDetection`. |
| `enable_waving_detection` | `SocialAdapter` | Subscribe to `ALWavingDetection`. |
| `enable_sonar` | `SonarAdapter` | Subscribe to `ALSonar`. |
| `face_match_max_angle_rad` | `FaceAdapter.match_faces_to_people` | Max yaw+pitch delta for matching face labels to people IDs. |
| `expression_labels` | `SocialAdapter._expression` | Labels corresponding to expression score indices. |

Server-side use of this metadata is documented in [`../server/detection-tracking-and-fusion.md`](../server/detection-tracking-and-fusion.md).

## `dialog` Section

This section controls language and dynamic concepts.

| Key | Meaning |
|---|---|
| `enable_dynamic_memory_concepts` | Allows `DialogAdapter.refresh_memory_concepts` to update ALDialog dynamic concepts. |
| `language` | `auto`, `czech`, or `english`. `auto` follows current TTS language. |
| `memory_objects_max` | Max object labels inserted into `memory_objects`. |
| `memory_attributes_max` | Max attribute names inserted into `memory_attributes`. |
| `memory_relations_max` | Max relation names inserted into `memory_relations`. |
| `memory_cached_questions_max` | Max pregenerated questions inserted into `memory_cached_questions`. |
| `refresh_after_detect` | Refresh concepts after quick look or visual refresh detects. |
| `refresh_after_scan` | Refresh concepts after scans. |
| `refresh_after_reset` | Refresh concepts after memory reset. |

`normalize_config` removes old `language_code`, normalizes language, and clamps max values to at least 1.

## `panorama` Section

This section controls full scan mode.

| Key | Meaning |
|---|---|
| `enabled` | If false, scan mode falls back to sequential detect. |
| `mode` | `panorama_detect` or `sequential_detect`. |
| `stick_together` | Passed to server panorama endpoint. If true, server stitches images. |
| `summary_after_scan` | If true, client asks server chat for a spoken scan summary. |
| `render_limit` | Legacy/default memory render limit if tablet setting is absent. |

`scan_planner.scan_mode` returns `panorama_detect` unless disabled or explicitly set to `sequential_detect`.

## `tablet` Section

This section controls memory display and fake tablet development.

| Key | Used by | Meaning |
|---|---|---|
| `memory_render_limit` | `scan_planner.memory_render_limit` | Number of cropped objects/rendered graph items requested from server summary. |
| `local_app_name` | `TabletAdapter` | Robot app name under `/apps/<name>/`. |
| `bridge_retry_attempts` | `TabletAdapter.push_memory_payload` | Page-ready and payload injection retries. |
| `bridge_retry_interval_seconds` | `TabletAdapter.push_memory_payload` | Delay between tablet JS retries. |
| `pregenerated_questions_count` | `TurnManager._pregenerated_questions_count` | Number of Q/A pairs requested for memory display. |
| `fake_tablet` | `_initialize_tablet_adapter` | Enables desktop fake tablet server. |
| `fake_host` | `FakeTabletAdapter` | Fake tablet bind host. |
| `fake_port` | `FakeTabletAdapter` | Fake tablet bind port. |
| `fake_poll_interval_ms` | Fake tablet URL and JS | Browser polling interval for `/payload.json`. |

The tablet architecture is documented in [`tablet-memory-ui.md`](tablet-memory-ui.md).

## Current Local Config Notes

The checked-in `client_config.json` currently enables:

- `capture.fake_camera=true` with a local desktop image folder.
- `tablet.fake_tablet=true` with port `8766`.
- `server.base_url` pointing at an ngrok URL.
- `auto_refresh_before_chat=false`.

Those values are useful for local development but may not be appropriate for robot deployment. For deployed robot use, usually set fake camera/tablet to false and configure a server URL reachable from the robot network.
