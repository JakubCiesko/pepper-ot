<?xml version="1.0" encoding="UTF-8" ?>
<Package name="pepper-grounded-client" format_version="4">
    <Manifest src="manifest.xml" />
    <BehaviorDescriptions>
        <BehaviorDescription name="behavior" src="testrun" xar="behavior.xar" />
    </BehaviorDescriptions>
    <Dialogs>
        <Dialog name="pepper-grounded-client" src="pepper-grounded-client/pepper-grounded-client.dlg" />
    </Dialogs>
    <Resources>
        <File name="peppergroundedclient" src="scripts/peppergroundedclient.py" />
        <File name="client_config" src="scripts/client_config.json" />
        <File name="stk_init" src="scripts/stk/__init__.py" />
        <File name="stk_events" src="scripts/stk/events.py" />
        <File name="stk_logging" src="scripts/stk/logging.py" />
        <File name="stk_runner" src="scripts/stk/runner.py" />
        <File name="stk_services" src="scripts/stk/services.py" />
        <File name="pc_init" src="scripts/pepper_client/__init__.py" />
        <File name="pc_core_init" src="scripts/pepper_client/core/__init__.py" />
        <File name="pc_session_store" src="scripts/pepper_client/core/session_store.py" />
        <File name="pc_transport" src="scripts/pepper_client/core/transport.py" />
        <File name="pc_turn_manager" src="scripts/pepper_client/core/turn_manager.py" />
        <File name="pc_interaction_init" src="scripts/pepper_client/interaction/__init__.py" />
        <File name="pc_dialog_adapter" src="scripts/pepper_client/interaction/dialog_adapter.py" />
        <File name="pc_speech_adapter" src="scripts/pepper_client/interaction/speech_adapter.py" />
        <File name="pc_speech_policy" src="scripts/pepper_client/interaction/speech_policy.py" />
        <File name="pc_tablet_adapter" src="scripts/pepper_client/interaction/tablet_adapter.py" />
        <File name="pc_perception_init" src="scripts/pepper_client/perception/__init__.py" />
        <File name="pc_camera_adapter" src="scripts/pepper_client/perception/camera_adapter.py" />
        <File name="pc_face_adapter" src="scripts/pepper_client/perception/face_adapter.py" />
        <File name="pc_people_adapter" src="scripts/pepper_client/perception/people_adapter.py" />
        <File name="pc_pose_adapter" src="scripts/pepper_client/perception/pose_adapter.py" />
        <File name="pc_robot_context" src="scripts/pepper_client/perception/robot_context.py" />
        <File name="pc_scan_planner" src="scripts/pepper_client/perception/scan_planner.py" />
        <File name="pc_social_adapter" src="scripts/pepper_client/perception/social_adapter.py" />
        <File name="pc_sonar_adapter" src="scripts/pepper_client/perception/sonar_adapter.py" />
        <File name="pc_utils_init" src="scripts/pepper_client/utils/__init__.py" />
        <File name="pc_config" src="scripts/pepper_client/utils/config.py" />
        <File name="pc_error_policy" src="scripts/pepper_client/utils/error_policy.py" />
        <File name="pc_ids" src="scripts/pepper_client/utils/ids.py" />
        <File name="pc_logging_utils" src="scripts/pepper_client/utils/logging.py" />
        <File name="pc_metadata_builder" src="scripts/pepper_client/utils/metadata_builder.py" />
        <File name="pc_text_utils" src="scripts/pepper_client/utils/text.py" />
        <File name="pc_time_utils" src="scripts/pepper_client/utils/timing.py" />
        <File name="translation_cs_CZ" src="translations/translation_cs_CZ.qm" />
        <File name="translation_en_US" src="translations/translation_en_US.qm" />
        <File name="index" src="html/index.html" />
        <File name="style" src="html/css/style.css" />
        <File name="state_js" src="html/js/state.js" />
        <File name="render_js" src="html/js/render.js" />
        <File name="service_bridge_js" src="html/js/service_bridge.js" />
        <File name="utils_js" src="html/js/utils.js" />
        <File name="app_js" src="html/js/app.js" />
        <File name="fake_tablet_js" src="html/js/fake_tablet.js" />
        <File name="icon" src="icon.png" />
    </Resources>
    <Topics>
        <Topic name="pepper-grounded-client_enu" src="pepper-grounded-client/pepper-grounded-client_enu.top" topicName="pepper-grounded-client" language="en_US" />
        <Topic name="pepper-grounded-client_czc" src="pepper-grounded-client/pepper-grounded-client_czc.top" topicName="pepper-grounded-client" language="cs_CZ" />
    </Topics>
    <IgnoredPaths>
        <Path src="scripts/pepper_client/interaction/__pycache__/speech_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/interaction/speech_policy.pyc" />
        <Path src="scripts/pepper_client/perception/__pycache__/scan_planner.cpython-312.pyc" />
        <Path src="scripts/pepper_client/perception/face_adapter.pyc" />
        <Path src="scripts/pepper_client/perception/social_adapter.pyc" />
        <Path src="scripts/pepper_client/interaction/__pycache__" />
        <Path src="scripts/pepper_client/perception/__pycache__/pose_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__" />
        <Path src="scripts/pepper_client/interaction/__pycache__/dialog_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/perception/scan_planner.pyc" />
        <Path src="scripts/pepper_client/utils/time_utils.pyc" />
        <Path src="scripts/pepper_client/core/__pycache__/__init__.cpython-312.pyc" />
        <Path src="scripts/pepper_client/perception/__pycache__/people_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/utils/__pycache__/config.cpython-312.pyc" />
        <Path src="scripts/pepper_client/interaction/__pycache__/tablet_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/perception/__pycache__/__init__.cpython-312.pyc" />
        <Path src="scripts/__pycache__/peppergroundedclient.cpython-312.pyc" />
        <Path src="scripts/stk/__init__.pyc" />
        <Path src="scripts/stk/__pycache__/runner.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/speech_policy.cpython-312.pyc" />
        <Path src="scripts/pepper_client/core/__init__.pyc" />
        <Path src="scripts/pepper_client/interaction/__init__.pyc" />
        <Path src="scripts/pepper_client/__pycache__/face_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/core/__pycache__/session_store.cpython-312.pyc" />
        <Path src="scripts/stk/__pycache__/services.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/sonar_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/core/__pycache__" />
        <Path src="scripts/pepper_client/__pycache__/metadata_builder.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/config.cpython-312.pyc" />
        <Path src="scripts/pepper_client/utils/__pycache__/ids.cpython-312.pyc" />
        <Path src="scripts/pepper_client/interaction/dialog_adapter.pyc" />
        <Path src="scripts/pepper_client/__pycache__/social_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/perception/__pycache__/camera_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/interaction/speech_adapter.pyc" />
        <Path src="scripts/stk/__pycache__" />
        <Path src="scripts/pepper_client/__pycache__/ids.cpython-312.pyc" />
        <Path src="scripts/pepper_client/interaction/tablet_adapter.pyc" />
        <Path src="scripts/pepper_client/perception/pose_adapter.pyc" />
        <Path src="scripts/pepper_client/utils/config.pyc" />
        <Path src="scripts/pepper_client/utils/ids.pyc" />
        <Path src="scripts/stk/__pycache__/logging.cpython-312.pyc" />
        <Path src="scripts/pepper_client/utils/metadata_builder.pyc" />
        <Path src="scripts/__pycache__" />
        <Path src="scripts/stk/services.pyc" />
        <Path src="scripts/pepper_client/utils/logging_utils.pyc" />
        <Path src="scripts/stk/logging.pyc" />
        <Path src="scripts/pepper_client/utils/__pycache__" />
        <Path src="scripts/pepper_client/utils/__pycache__/metadata_builder.cpython-312.pyc" />
        <Path src="scripts/pepper_client/interaction/__pycache__/speech_policy.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/pose_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/core/__pycache__/turn_manager.cpython-312.pyc" />
        <Path src="scripts/pepper_client/interaction/__pycache__/__init__.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/time_utils.cpython-312.pyc" />
        <Path src="scripts/pepper_client/perception/__pycache__/face_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/utils/error_policy.pyc" />
        <Path src="scripts/stk/runner.pyc" />
        <Path src="scripts/pepper_client/utils/__pycache__/logging_utils.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/scan_planner.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/__init__.cpython-312.pyc" />
        <Path src="scripts/pepper_client/core/session_store.pyc" />
        <Path src="scripts/pepper_client/perception/__pycache__/robot_context.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/logging_utils.cpython-312.pyc" />
        <Path src="scripts/pepper_client/perception/sonar_adapter.pyc" />
        <Path src="scripts/pepper_client/utils/text.pyc" />
        <Path src="scripts/pepper_client/perception/__init__.pyc" />
        <Path src="scripts/pepper_client/perception/robot_context.pyc" />
        <Path src="scripts/pepper_client/utils/__pycache__/error_policy.cpython-312.pyc" />
        <Path src="scripts/pepper_client/utils/__pycache__/__init__.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/people_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/utils/timing.pyc" />
        <Path src="scripts/pepper_client/perception/__pycache__" />
        <Path src="scripts/pepper_client/perception/__pycache__/social_adapter.cpython-312.pyc" />
        <Path src="scripts/stk/__pycache__/__init__.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__init__.pyc" />
        <Path src="scripts/pepper_client/utils/logging.pyc" />
        <Path src="scripts/pepper_client/perception/people_adapter.pyc" />
        <Path src="scripts/pepper_client/__pycache__/speech_adapter.cpython-312.pyc" />
        <Path src="pepper-grounded-client/system_enu.top" />
        <Path src="Makefile" />
        <Path src="scripts/pepper_client/__pycache__/error_policy.cpython-312.pyc" />
        <Path src="scripts/pepper_client/core/transport.pyc" />
        <Path src="scripts/pepper_client/core/turn_manager.pyc" />
        <Path src="scripts/stk/__pycache__/events.cpython-312.pyc" />
        <Path src="scripts/pepper_client/utils/__init__.pyc" />
        <Path src="scripts/pepper_client/utils/__pycache__/time_utils.cpython-312.pyc" />
        <Path src="scripts/pepper_client/perception/__pycache__/sonar_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/tablet_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/core/__pycache__/transport.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/camera_adapter.cpython-312.pyc" />
        <Path src="scripts/pepper_client/__pycache__/robot_context.cpython-312.pyc" />
        <Path src="scripts/pepper_client/perception/camera_adapter.pyc" />
    </IgnoredPaths>
    <Translations auto-fill="en_US">
        <Translation name="translation_cs_CZ" src="translations/translation_cs_CZ.ts" language="cs_CZ" />
        <Translation name="translation_en_US" src="translations/translation_en_US.ts" language="en_US" />
    </Translations>
</Package>
