"""
A sample showing how to have a NAOqi service as a Python app.
"""

__version__ = "0.0.3"

__copyright__ = "Copyright 2015, Aldebaran Robotics"
__author__ = "Jakub Ciesko"
__email__ = "jakub.ciesko@gmail.com"


import io
import time

from conversation import Conversation
from PIL import Image
import qi
import requests
import stk.events
import stk.logging
import stk.runner
import stk.services


class PepperObjectRecognition:

    APP_ID = "com.aldebaran.PepperObjectRecognition"

    def __init__(self, qiapp):
        # logger setup
        self.logger = stk.logging.get_logger(qiapp.session, self.APP_ID)

        self.qiapp = qiapp
        self.session = qiapp.session
        #self.memory = self.session.service("ALMemory")
        self.tts = self.session.service("ALTextToSpeech")
        self.video = self.session.service("ALVideoDevice")

        # Config of connection
        self.server_base_url = "https://raye-uncontractile-attently.ngrok-free.dev"
        self.server_url = self.server_base_url + "/api/detect"
        self.conf_threshold = 0.3

        # Initialize Logic
        self.conversation = Conversation(memory_length=50, language="cs")
        self.camera_handle = None

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def detect(self, lang_code):
        """
        Called by Dialog.
        lang_code: 'en' or 'cs'
        """
        self.logger.info("Dialog triggered detection in lang: %s", lang_code)
        self.conversation.language = lang_code
        self.detect_and_speak()

    def detect_and_speak(self):
        # Subscribe to camera temporarily
        camera_name = self.APP_ID + "_" + str(time.time())
        self.logger.info("Starting camera called: %s, resolution 640x480, RGB color, 10 FPS", camera_name)
        self.camera_handle = self.video.subscribeCamera(camera_name, 0, 2, 11, 10)
        self.logger.info("Camera started")

        try:
            self.logger.info("Taking picture...")
            nao_img = self.video.getImageRemote(self.camera_handle)

            if nao_img is None:
                self.logger.warn("No picture taken.")
                self.tts.say(self.conversation.no_data_message())
                return

            img_bytes = self._process_nao_image(nao_img)
            data = self.get_processed_data_from_server(img_bytes)
            self.handle_processed_data(data)

        finally:
            # Freeing up the camera
            if self.camera_handle:
                self.video.unsubscribe(self.camera_handle)
                self.camera_handle = None



    def _process_nao_image(self, nao_img):
        width = nao_img[0]
        height = nao_img[1]
        data = nao_img[6]
        self.logger.info("Processing image (%dx%d)", width, height)
        image = Image.frombytes("RGB", (width, height), str(data))
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_jpeg_bytes = img_buffer.getvalue()
        return img_jpeg_bytes

    def get_processed_data_from_server(self, img_jpeg_bytes):
        data = {}
        try:
            files = {
                "image": ("capture.jpg", img_jpeg_bytes, "image/jpeg")
            }
            response = requests.post(self.server_url, files=files, timeout=5)
            if response.status_code == 200:
                data = response.json()
            else:
                self.logger.warn("Server returned %d", response.status_code)
        except Exception, e:
            self.logger.error("Network Request failed: %s", str(e))
        return data

    def send_sentence_to_server(self, sentence):
        try:
            payload = {"sentence": sentence}
            response = requests.post(
                self.server_base_url + "/dashboard/sentence",
                json=payload,
                timeout=3
            )
            if response.status_code != 200:
                self.logger.warn("Dashboard server returned %d", response.status_code)
        except Exception as e:
            self.logger.error("Failed sending sentence to dashboard: %s", str(e))

    def handle_processed_data(self, data):
        objects = data.get("objects", [])
        if not objects:
            self.logger.info("No objects detected.")
            sentence = self.conversation.observe([])  # this lets the robot comment on lack of input :)
            self.tts.say(sentence)
            return

        labels = [
            obj["label"]
            for obj in objects
            # if obj.get("confidence", 0) > self.conf_threshold  # offloaded to server
        ]
        if not labels:
            self.logger.info("No labels for objects detected.")
            return

        sentence = self.conversation.observe(labels)
        self.logger.info("[ROBOT]: %s", sentence)
        self.tts.say(sentence)
        self.send_sentence_to_server(sentence)

    @qi.bind(returnType=qi.Void, paramsType=[])
    def stop(self):
        "Stop the service."
        self.logger.info("PepperObjectRecognition stopped by user request.")
        self.qiapp.stop()

    def cleanup(self):
        self.logger.info("Cleaning up")
        if self.camera_handle:
            self.video.unsubscribe(self.camera_handle)
        self.logger.info("Cleaned up")

    @qi.nobind
    def on_stop(self):
        "Cleanup (add yours if needed)"
        self.logger.info("Cleaning up camera.")
        self.cleanup()
        self.logger.info("PepperObjectRecognition finished.")

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def say(self, text):
        self.logger.info("Dialog called say(): %s", text)
        self.tts.say(text)


####################
# Setup and Run
####################

if __name__ == "__main__":
    stk.runner.run_service(PepperObjectRecognition)
