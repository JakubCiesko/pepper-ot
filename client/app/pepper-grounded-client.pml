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
        <File name="icon" src="icon.png" />
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
        <File name="pc_time_utils" src="scripts/pepper_client/utils/timing.py" />
        <File name="pc_text_utils" src="scripts/pepper_client/utils/text.py" />
        <File name="translation_cs_CZ" src="translations/translation_cs_CZ.qm" />
        <File name="translation_en_US" src="translations/translation_en_US.qm" />
        <File name="index" src="html/index.html" />
    </Resources>
    <Topics>
        <Topic name="pepper-grounded-client_enu" src="pepper-grounded-client/pepper-grounded-client_enu.top" topicName="pepper-grounded-client" language="en_US" />
        <Topic name="pepper-grounded-client_czc" src="pepper-grounded-client/pepper-grounded-client_czc.top" topicName="pepper-grounded-client" language="cs_CZ" />
    </Topics>
    <IgnoredPaths />
    <Translations auto-fill="en_US">
        <Translation name="translation_cs_CZ" src="translations/translation_cs_CZ.ts" language="cs_CZ" />
        <Translation name="translation_en_US" src="translations/translation_en_US.ts" language="en_US" />
    </Translations>
</Package>
