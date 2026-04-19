# -*- coding: utf-8 -*-
__version__ = "0.0.3"

__copyright__ = "Copyright 2015, Aldebaran Robotics"
__author__ = "Jakub Ciesko"
__email__ = "jakub.ciesko@gmail.com"

import json
import os
import random

import qi

import stk.logging
import stk.runner
import stk.services
from pepper_client.core.session_store import SessionStore
from pepper_client.core.transport import PepperServerTransport
from pepper_client.core.turn_manager import TurnManager
from pepper_client.interaction import speech_policy
from pepper_client.interaction.dialog_adapter import DialogAdapter
from pepper_client.interaction.speech_adapter import SpeechAdapter
from pepper_client.interaction.tablet_adapter import (FakeTabletAdapter,
                                                      TabletAdapter)
from pepper_client.perception.camera_adapter import (CameraAdapter,
                                                     FakeCameraAdapter)
from pepper_client.perception.face_adapter import FaceAdapter
from pepper_client.perception.people_adapter import PeopleAdapter
from pepper_client.perception.pose_adapter import PoseAdapter
from pepper_client.perception.robot_context import RobotContextCollector
from pepper_client.perception.social_adapter import SocialAdapter
from pepper_client.perception.sonar_adapter import SonarAdapter
from pepper_client.utils import config as client_config
from pepper_client.utils import text as text_utils
from pepper_client.utils.metadata_builder import MetadataBuilder


class PepperGroundedClient(object):
    APP_ID = "PepperGroundedClient"

    def __init__(self, qiapp):
        self.qiapp = qiapp
        self.session = qiapp.session
        self.logger = stk.logging.get_logger(self.session, self.APP_ID)
        self.services = stk.services.ServiceCache(self.session)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = client_config.build_script_path(
            self.script_dir, "client_config.json")
        self.config = client_config.load_config(self.config_path, self.logger)
        self.logger.info("Loaded client config from %s", self.config_path)

        self.session_store = SessionStore(self.logger)
        self.session_store.set_server_base_url(
            self.config["server"].get("base_url"))

        self.transport = PepperServerTransport(self.config, self.logger)
        self.camera_adapter = None
        self._initialize_camera_adapter()
        self.pose_adapter = PoseAdapter(self.services, self.logger)
        self.face_adapter = FaceAdapter(self.services, self.config,
                                        self.logger)
        self.people_adapter = PeopleAdapter(self.services, self.config,
                                            self.logger)
        self.social_adapter = SocialAdapter(
            self.services,
            self.config,
            self.logger,
            self.face_adapter,
        )
        self.sonar_adapter = SonarAdapter(self.services, self.config,
                                          self.logger)
        self.speech_adapter = SpeechAdapter(self.services, self.config,
                                            self.logger)
        self.tablet_adapter = None
        self._initialize_tablet_adapter()
        self.dialog_adapter = DialogAdapter(self.services, self.config,
                                            self.logger)
        self.robot_context = RobotContextCollector(
            self.pose_adapter,
            self.people_adapter,
            self.social_adapter,
            self.sonar_adapter,
            self.logger,
        )
        self.metadata_builder = MetadataBuilder(self.logger)
        self.turn_manager = TurnManager(
            self.config,
            self.session_store,
            self.camera_adapter,
            self.pose_adapter,
            self.robot_context,
            self.metadata_builder,
            self.transport,
            self.speech_adapter,
            self.tablet_adapter,
            self.dialog_adapter,
            self.logger,
        )

    @qi.nobind
    def _initialize_camera_adapter(self):
        if self.config["capture"].get("fake_camera",
                                      False) and self.config["capture"].get(
                                          "fake_camera_path", None):
            self.logger.info(
                "Using fake camera for testing and simulation, path: %s",
                self.config["capture"]["fake_camera_path"])
            self.camera_adapter = FakeCameraAdapter(
                self.config["capture"].get("fake_camera_path"), self.logger)
        else:
            self.logger.info("Using default robot camera")
            self.camera_adapter = CameraAdapter(self.services, self.config,
                                                self.logger)

    @qi.nobind
    def _initialize_tablet_adapter(self):
        if self.config.get("tablet", {}).get("fake_tablet", False):
            self.tablet_adapter = FakeTabletAdapter(self.services, self.config,
                                                    self.logger)
            fake_url = self.tablet_adapter.local_fake_url()
            self.logger.info(
                "Using fake tablet adapter for local browser mirror, url: %s" %
                fake_url)

        else:
            self.logger.info("Using robot ALTabletService adapter")
            self.tablet_adapter = TabletAdapter(self.services, self.config,
                                                self.logger)

    @qi.nobind
    def on_start(self):
        self.logger.info(
            "PepperGroundedClient starting with server=%s dialog_language=%s",
            self.config["server"].get("base_url"),
            self.config.get("dialog", {}).get("language"),
        )
        self.face_adapter.start()
        self.robot_context.start()
        self.turn_manager.refresh_memory_concepts()

    @qi.nobind
    def on_stop(self):
        self.logger.info("PepperGroundedClient stopping")
        self.tablet_adapter.hide_memory_page()
        self.turn_manager.shutdown()
        self.robot_context.stop()
        self.face_adapter.stop()

    @qi.nobind
    def _runtime_language(self, requested_lang=None):
        return speech_policy.resolve_language_state(
            self.config,
            requested=requested_lang,
            tts=self.speech_adapter.tts,
            logger=self.logger,
        )[1]

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def look(self, lang_code):
        self.logger.info("look called lang_code=%s", lang_code)
        self.turn_manager.start_look(lang_code)

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def scan(self, lang_code):
        self.logger.info("scan called lang_code=%s", lang_code)
        self.turn_manager.start_scan(lang_code)

    @qi.bind(returnType=qi.Void, paramsType=[qi.String, qi.String])
    def ask(self, lang_code, query):
        self.logger.info("ask called lang_code=%s query=%s", lang_code, query)
        self.turn_manager.start_ask(lang_code, query, force_refresh=False)

    @qi.bind(returnType=qi.Void, paramsType=[qi.String, qi.String])
    def refreshAndAsk(self, lang_code, query):
        self.logger.info("refreshAndAsk called lang_code=%s query=%s",
                         lang_code, query)
        self.turn_manager.start_ask(lang_code, query, force_refresh=True)

    @qi.bind(returnType=qi.Void, paramsType=[])
    def resetConversation(self):
        self.logger.info("resetConversation called")
        chat_id = self.session_store.get_chat_id()
        self.session_store.reset_conversation()
        if chat_id:
            try:
                self.transport.reset_conversation(chat_id)
            except Exception as exc:
                self.logger.info("Server conversation reset failed for %s: %s",
                                 chat_id, exc)
        reset_lang = speech_policy.resolve_language_state(
            self.config,
            tts=self.speech_adapter.tts,
            logger=self.logger,
        )[1]
        self.speech_adapter.say(
            speech_policy.acknowledgement("reset", reset_lang),
            self.config.get("dialog", {}).get("language", "auto"),
        )

    @qi.bind(returnType=qi.Void, paramsType=[qi.String, qi.String])
    def askAboutObject(self, lang_code, object_label):
        self.logger.info(
            "askAboutObject called lang_code=%s object_label=%s",
            lang_code,
            object_label,
        )
        self.turn_manager.start_object_ask(lang_code, object_label)

    @qi.bind(returnType=qi.Void, paramsType=[qi.String, qi.String])
    def answerCachedQuestion(self, lang_code, question):
        self.logger.info(
            "answerCachedQuestion called lang_code=%s question=%s",
            lang_code,
            question,
        )
        self.turn_manager.start_cached_answer(lang_code, question)

    # @qi.bind(returnType=qi.List)
    # def listCachedAnswers(self):
    #     return self.session_store.get_cached_answers()

    @qi.nobind
    def prefix_for_listing(self, lang_code, empty):
        lang_code = self._runtime_language(lang_code)
        if empty:
            if lang_code == "cs":
                return "Je mi to líto ale nic nevidím"
            return "I am sorry, but I don't see anything"
        if lang_code == "cs":
            return "Vidím "
        return "I see "

    @qi.nobind
    def listDynamicConcept(self,
                           lang_code,
                           concept,
                           sample_size,
                           return_concept_only=False):
        self.logger.info("listDynamicConcept called lang_code=%s concept=%s",
                         lang_code, concept)
        concept_getters = {
            "objects": self.session_store.get_memory_labels,
            "attributes": self.session_store.get_memory_attributes,
            "relations": self.session_store.get_memory_relations,
            "cached_questions": self.session_store.get_cached_questions,
            "cached_answers": self.session_store.get_cached_answers,
        }
        getter = concept_getters.get(concept)
        error_output = self.prefix_for_listing(lang_code, True)
        if not getter:
            return error_output
        dynamic_concept = getter()
        if not dynamic_concept:
            return error_output if not return_concept_only else dynamic_concept

        if return_concept_only:
            return dynamic_concept
        # all are lists but just to be sure
        if not isinstance(dynamic_concept, (list, tuple)):
            return error_output

        if sample_size > 0:
            sample = random.sample(dynamic_concept,
                                   min(sample_size, len(dynamic_concept)))
        else:
            sample = dynamic_concept

        prefix = self.prefix_for_listing(lang_code, False)
        prefix = text_utils.clean_text_unicode(prefix) + u" "
        sample = [text_utils.clean_text_unicode(s) for s in sample]
        return prefix + u", ".join([s.replace(u"_", u" ") for s in sample])

    # TODO: make the 10 tunable
    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def listRelations(self, lang_code):
        self.say(self.listDynamicConcept(lang_code, "relations", 10))

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def listAttributes(self, lang_code):
        self.say(self.listDynamicConcept(lang_code, "attributes", 10))

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def listObjects(self, lang_code):
        self.say(self.listDynamicConcept(lang_code, "objects", 10))

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def listCachedQuestions(self, lang_code):
        questions = self.listDynamicConcept(lang_code,
                                            "cached_questions",
                                            1,
                                            return_concept_only=True)
        questions = random.sample(questions, min(len(questions), 1))
        lang_code = self._runtime_language(lang_code)
        if questions:
            if lang_code == "cs":
                text = u"Mužeš se zeptat třeba tohle: %s" % questions[0]
            else:
                text = u"You can ask me like this: %s" % questions[0]
        else:
            if lang_code == "cs":
                text = u"Zeptej se cokoliv, teď nemám nic přichystané"
            else:
                text = u"You can ask anything you like. I have nothing prepared"
        self.say(text_utils.clean_text_unicode(text))

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def showMemory(self, lang_code):
        self.logger.info("showMemory called lang_code=%s", lang_code)
        self.turn_manager.start_show_memory(lang_code)

    @qi.bind(returnType=qi.Void, paramsType=[])
    def hideMemory(self):
        self.logger.info("hideMemory called")
        self.tablet_adapter.hide_memory_page()

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def resetMemory(self, lang_code):
        self.logger.info("resetMemory called lang_code=%s", lang_code)
        self.turn_manager.start_reset_memory(lang_code)

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def refreshMemoryConcepts(self, lang_code=None):
        self.logger.info("refreshMemoryConcepts called")
        self.turn_manager.refresh_memory_concepts(lang_code=lang_code)

    #TODO: this should not be used anywhere really lots of binded functions are useless now
    @qi.bind(returnType=qi.String, paramsType=[qi.String])
    def setDialogLanguage(self, mode):
        normalized = speech_policy.normalize_dialog_language(mode)
        self.logger.info(
            "setDialogLanguage called mode=%s normalized=%s",
            mode,
            normalized,
        )
        self.config.setdefault("dialog", {})["language"] = normalized
        client_config.save_config(self.config, self.logger)
        return normalized

    @qi.bind(returnType=qi.String, paramsType=[qi.String])
    def setServerBaseUrl(self, base_url):
        base_url = str(base_url or "").strip().rstrip("/")
        self.logger.info("setServerBaseUrl called base_url=%s", base_url)
        self.config["server"]["base_url"] = base_url
        self.session_store.set_server_base_url(base_url)
        self.transport.update_config(self.config)
        client_config.save_config(self.config, self.logger)
        return base_url

    @qi.bind(returnType=qi.String, paramsType=[])
    def reloadConfig(self):
        self.logger.info("reloadConfig called")
        loaded = client_config.load_config(self.config_path, self.logger)
        self._apply_loaded_config(loaded)
        return json.dumps(self.getStatusObject(), sort_keys=True)

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def say(self, text):
        text = text_utils.clean_text_unicode(text)
        self.logger.info("say called text=%s", text.encode("utf-8"))
        self.speech_adapter.say(text)

    @qi.bind(returnType=qi.String, paramsType=[])
    def getStatus(self):
        return json.dumps(self.getStatusObject(), sort_keys=True)

    @qi.bind(returnType=qi.Void, paramsType=[])
    def stop(self):
        self.logger.info("stop called")
        self.qiapp.stop()

    @qi.nobind
    def getStatusObject(self):
        status = self.turn_manager.status()
        status["server_base_url"] = self.config["server"].get("base_url")
        status["dialog_language"] = self.config.get("dialog",
                                                    {}).get("language")
        return status

    @qi.nobind
    def _apply_loaded_config(self, loaded):
        self.robot_context.stop()
        self.face_adapter.stop()
        self.config.clear()
        self.config.update(loaded)
        self.session_store.set_server_base_url(
            self.config["server"].get("base_url"))
        self.transport.update_config(self.config)
        self.dialog_adapter.update_config(self.config)
        self.face_adapter.start()
        self.robot_context.start()
        self._initialize_camera_adapter()
        self._initialize_tablet_adapter()
        if self.turn_manager:
            self.turn_manager.camera_adapter = self.camera_adapter
            self.turn_manager.tablet_adapter = self.tablet_adapter
        self.logger.info("New config: %s", self.config)
        self.logger.info("Client config reloaded")


if __name__ == "__main__":
    run_local = True
    czech = True
    if run_local:
        app = qi.Application()
        app.start()

        session = app.session

        service_instance = PepperGroundedClient(app)

        session.registerService("PepperGroundedClient", service_instance)

        service_instance.on_start()
        if czech:
            dialog = session.service("ALDialog")
            dialog.setLanguage("Czech")
            #asr = session.service("ALSpeechRecognition")
            #asr.setLanguage("Czech")
            tts = session.service("ALTextToSpeech")
            tts.setLanguage("Czech")
        app.run()

        service_instance.on_stop()
    else:
        stk.runner.run_service(PepperGroundedClient)
