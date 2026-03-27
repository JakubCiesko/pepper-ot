__version__ = "0.0.3"

__copyright__ = "Copyright 2015, Aldebaran Robotics"
__author__ = "Jakub Ciesko"
__email__ = "jakub.ciesko@gmail.com"

import json
import os

import qi

from pepper_client.utils import config as client_config
from pepper_client.interaction import speech_policy
from pepper_client.perception.camera_adapter import CameraAdapter, FakeCameraAdapter
from pepper_client.perception.face_adapter import FaceAdapter
from pepper_client.utils.metadata_builder import MetadataBuilder
from pepper_client.perception.people_adapter import PeopleAdapter
from pepper_client.perception.pose_adapter import PoseAdapter
from pepper_client.perception.robot_context import RobotContextCollector
from pepper_client.core.session_store import SessionStore
from pepper_client.perception.social_adapter import SocialAdapter
from pepper_client.perception.sonar_adapter import SonarAdapter
from pepper_client.interaction.speech_adapter import SpeechAdapter
from pepper_client.interaction.tablet_adapter import TabletAdapter
from pepper_client.core.transport import PepperServerTransport
from pepper_client.core.turn_manager import TurnManager
import stk.logging
import stk.runner
import stk.services


class PepperGroundedClient(object):
    APP_ID = "com.aldebaran.PepperGroundedClient"

    def __init__(self, qiapp):
        self.qiapp = qiapp
        self.session = qiapp.session
        self.logger = stk.logging.get_logger(self.session, self.APP_ID)
        self.services = stk.services.ServiceCache(self.session)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = client_config.build_script_path(self.script_dir, "client_config.json")
        self.config = client_config.load_config(self.config_path, self.logger)
        self.logger.info("Loaded client config from %s", self.config_path)

        self.session_store = SessionStore(self.logger)
        self.session_store.set_output_language_mode(
            self.config["language"].get("output_language_mode", "default")
        )
        self.session_store.set_server_base_url(self.config["server"].get("base_url"))

        self.transport = PepperServerTransport(self.config, self.logger)
        self.camera_adapter = None
        self._initialize_camera_adapter()
        self.pose_adapter = PoseAdapter(self.services, self.logger)
        self.face_adapter = FaceAdapter(self.services, self.config, self.logger)
        self.people_adapter = PeopleAdapter(self.services, self.config, self.logger)
        self.social_adapter = SocialAdapter(
            self.services,
            self.config,
            self.logger,
            self.face_adapter,
        )
        self.sonar_adapter = SonarAdapter(self.services, self.config, self.logger)
        self.speech_adapter = SpeechAdapter(self.services, self.logger)
        self.tablet_adapter = TabletAdapter(self.services, self.config, self.logger)
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
            self.logger,
        )

    @qi.nobind
    def _initialize_camera_adapter(self):
        if self.config["capture"].get("fake_camera", False) and self.config["capture"].get("fake_camera_path", None):
            self.logger.info("Using fake camera for testing and simulation, path: %s", self.config["capture"]["fake_camera_path"])
            self.camera_adapter = FakeCameraAdapter(self.config["capture"].get("fake_camera_path"), self.logger)
        else:
            self.logger.info("Using default robot camera")
            self.camera_adapter=CameraAdapter(self.services, self.config, self.logger)

    @qi.nobind
    def on_start(self):
        self.logger.info(
            "PepperGroundedClient starting with server=%s output_language=%s",
            self.config["server"].get("base_url"),
            self.config["language"].get("output_language_mode"),
        )
        self.face_adapter.start()
        self.robot_context.start()
        if self.config["behavior"].get("show_dashboard_on_start", False):
            self.tablet_adapter.show_dashboard()

    @qi.nobind
    def on_stop(self):
        self.logger.info("PepperGroundedClient stopping")
        self.turn_manager.shutdown()
        self.robot_context.stop()
        self.face_adapter.stop()

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
        self.logger.info("refreshAndAsk called lang_code=%s query=%s", lang_code, query)
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
                self.logger.info("Server conversation reset failed for %s: %s", chat_id, exc)
        reset_lang = speech_policy.language_code(
            self.config["language"].get("default_dialog_language", "en")
        )
        self.speech_adapter.say(
            speech_policy.acknowledgement("reset", reset_lang),
            reset_lang,
        )

    @qi.bind(returnType=qi.String, paramsType=[qi.String])
    def setOutputLanguage(self, mode):
        normalized = client_config.normalize_output_language(mode)
        self.logger.info("setOutputLanguage called mode=%s normalized=%s", mode, normalized)
        self.transport.patch_output_language(normalized)
        self.session_store.set_output_language_mode(normalized)
        self.config["language"]["output_language_mode"] = normalized
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

    @qi.bind(returnType=qi.Void, paramsType=[])
    def showDashboard(self):
        self.logger.info("showDashboard called")
        self.tablet_adapter.show_dashboard()

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def say(self, text):
        self.logger.info("say called text=%s", text)
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
        status["dashboard_url"] = self.config["server"].get("dashboard_url")
        status["output_language_mode"] = self.config["language"].get("output_language_mode")
        return status

    @qi.nobind
    def _apply_loaded_config(self, loaded):
        self.robot_context.stop()
        self.face_adapter.stop()
        self.config.clear()
        self.config.update(loaded)
        self.session_store.set_output_language_mode(
            self.config["language"].get("output_language_mode", "default")
        )
        self.session_store.set_server_base_url(self.config["server"].get("base_url"))
        self.transport.update_config(self.config)
        self.face_adapter.start()
        self.robot_context.start()
        self._initialize_camera_adapter()
        if self.turn_manager:
            self.turn_manager.camera_adapter = self.camera_adapter
        self.logger.info("New config: %s", self.config)
        self.logger.info("Client config reloaded")


if __name__ == "__main__":
    stk.runner.run_service(PepperGroundedClient)
