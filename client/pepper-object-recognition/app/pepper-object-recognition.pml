<?xml version="1.0" encoding="UTF-8" ?>
<Package name="pepper-object-recognition" format_version="4">
    <Manifest src="manifest.xml" />
    <BehaviorDescriptions>
        <BehaviorDescription name="behavior" src="testrun" xar="behavior.xar" />
    </BehaviorDescriptions>
    <Dialogs>
        <Dialog name="pepper-object-recognition" src="pepper-object-recognition/pepper-object-recognition.dlg" />
    </Dialogs>
    <Resources>
        <File name="icon" src="icon.png" />
        <File name="pepperobjectrecognition" src="scripts/pepperobjectrecognition.py" />
        <File name="__init__" src="scripts/stk/__init__.py" />
        <File name="events" src="scripts/stk/events.py" />
        <File name="logging" src="scripts/stk/logging.py" />
        <File name="runner" src="scripts/stk/runner.py" />
        <File name="services" src="scripts/stk/services.py" />
        <File name="conversation" src="scripts/conversation.py" />
        <File name="conversation" src="scripts/conversation.pyc" />
        <File name="conversation_patterns" src="scripts/conversation_patterns.py" />
        <File name="__init__" src="scripts/stk/__init__.pyc" />
        <File name="events" src="scripts/stk/events.pyc" />
        <File name="logging" src="scripts/stk/logging.pyc" />
        <File name="runner" src="scripts/stk/runner.pyc" />
        <File name="services" src="scripts/stk/services.pyc" />
    </Resources>
    <Topics>
        <Topic name="pepper-object-recognition_czc" src="pepper-object-recognition/pepper-object-recognition_czc.top" topicName="pepper-object-recognition" language="cs_CZ" />
        <Topic name="pepper-object-recognition_enu" src="pepper-object-recognition/pepper-object-recognition_enu.top" topicName="pepper-object-recognition" language="en_US" />
    </Topics>
    <IgnoredPaths>
        <Path src=".metadata" />
    </IgnoredPaths>
    <Translations auto-fill="en_US">
        <Translation name="translation_cs_CZ" src="translations/translation_cs_CZ.ts" language="cs_CZ" />
        <Translation name="translation_en_US" src="translations/translation_en_US.ts" language="en_US" />
    </Translations>
</Package>
