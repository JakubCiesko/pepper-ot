# -*- coding: UTF-8 -*-

import logging
import time

from conversation_patterns import get_number_word
from conversation_patterns import get_random_phrase
from conversation_patterns import join_labels

logger = logging.getLogger(__name__)


class Conversation:
    def __init__(self, memory_length=10, language="en"):
        self.memory_length = memory_length
        self.language = language
        self.inter_frame_memory = {}  # {label: {"frames_seen": .., "last_seen": ....}}

    def no_data_message(self):
        return get_random_phrase(self.language, "NO_DATA")

    def _clean_label(self, label):
        """Ensures labels are consistent UTF-8 encoded strings for dict keys."""
        if isinstance(label, unicode):
            return label.encode("utf-8")
        return str(label)

    def observe_current_frame(self, clean_labels, now):
        newly_seen = []

        for label in clean_labels:
            if label not in self.inter_frame_memory:
                # first time ever
                self.inter_frame_memory[label] = {
                    "frames_seen": 1,
                    "last_seen": now,
                }
                newly_seen.append(label)
            else:
                # seen before: increment frames_seen
                self.inter_frame_memory[label]["frames_seen"] += 1
                self.inter_frame_memory[label]["last_seen"] = now
        logger.info("The robot sees for the first time: %s", str(newly_seen))
        return newly_seen

    def forget(self, now):
        logger.info(
            "Looking for labels to forget, memory length: %s", self.memory_length
        )
        forget_before = now - self.memory_length
        to_delete = []

        for label, info in self.inter_frame_memory.items():
            if info["last_seen"] < forget_before:
                to_delete.append(label)

        logger.info("The robot is going to forget %s", str(to_delete))
        logger.info("Memory before deletion: %s", str(self.inter_frame_memory))
        for label in to_delete:
            del self.inter_frame_memory[label]

        logger.info("Memory after deletion: %s", str(self.inter_frame_memory))

    def observe(self, labels):
        now = time.time()
        clean_labels = [self._clean_label(label) for label in labels]
        # forget
        self.forget(now)
        # observe
        newly_seen = self.observe_current_frame(clean_labels, now)
        # if newly_seen:
        #     # if something new seen, talk about that only ?
        #     return self.get_sentence(newly_seen)
        # else:
        #     return self.get_sentence(clean_labels)
        return self.get_sentence(clean_labels)

    def _get_label_counts(self, label_list):
        label_counts = {}
        for label in label_list:
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

    def get_sentence(self, labels):
        if not labels:
            return get_random_phrase(self.language, "NO_DATA")
        label_counts = self._get_label_counts(labels)
        unique_labels = list(label_counts.keys())
        # how many times now, how many frames already
        get_label_data = lambda label: (
            label_counts[label],
            self.inter_frame_memory.get(label, {"frames_seen": 0})["frames_seen"],
        )
        get_phrase = lambda phrase_type: get_random_phrase(
            language=self.language, phrase_type=phrase_type
        )
        # single unique label
        if len(unique_labels) == 1:
            label = unique_labels[0]
            n_obj_in_frame, obj_in_n_frames = get_label_data(label)

            if obj_in_n_frames <= 1:
                # first time
                if n_obj_in_frame == 1:
                    pattern = get_phrase("SINGLE_FIRST_TIME")
                else:
                    pattern = get_phrase("SINGLE_FIRST_TIME_MULTIPLE")
            elif obj_in_n_frames == 2:
                if n_obj_in_frame == 1:
                    pattern = get_phrase("SINGLE_SECOND_TIME")
                else:
                    pattern = get_phrase("SINGLE_AGAIN_MULTIPLE")
            else:
                if n_obj_in_frame == 1:
                    pattern = get_phrase("SINGLE_AGAIN")
                else:
                    pattern = get_phrase("SINGLE_AGAIN_MULTIPLE")

            return pattern.format(
                obj=label,
                count=get_number_word(self.language, "cardinal", n_obj_in_frame),
                count_word=get_number_word(self.language, "ordinal", obj_in_n_frames),
            )
        # more unique labels

        all_new = True
        all_known = True
        per_label_data = {}

        for label in unique_labels:
            n_obj_in_frame, obj_in_n_frames = get_label_data(label)
            per_label_data[label] = (n_obj_in_frame, obj_in_n_frames)

            if obj_in_n_frames <= 1:
                # new
                all_known = False
            else:
                # known
                all_new = False
        joined_objs = join_labels(unique_labels, self.language)
        if all_new:
            pattern = get_phrase("MULTI_ALL_NEW")
            return pattern.format(objs=joined_objs)

        if all_known:
            pattern = get_phrase("MULTI_ALL_KNOWN")
            return pattern.format(objs=joined_objs)

        pattern = get_phrase("MULTI_MIX")
        base_sentence = pattern.format(objs=joined_objs)
        details = []
        for label, (count_in_frame, frames_seen) in per_label_data.items():
            if frames_seen > 1:  # only comment on known ones
                detail_pattern = get_phrase("MULTI_DETAIL")
                details.append(
                    detail_pattern.format(
                        obj=label,
                        count_word=get_number_word(
                            self.language, "ordinal", frames_seen
                        ),
                    )
                )
                # just one extra
                break
        if details:
            return base_sentence + " " + " ".join(details)
        return base_sentence
