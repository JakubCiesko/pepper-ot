# -*- coding: UTF-8 -*-
# Number-word mapping (up to 5), fallback for larger
import random

NUMBER_WORDS = {
    "en": {
        "cardinal": {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            "many": "many",
        },
        "ordinal": {
            0: "zeroth",
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            "many": "manyth",
        },
    },
    "cs": {
        "cardinal": {
            0: "nula",
            1: "jedna",
            2: "dva",
            3: "tři",
            4: "čtyři",
            5: "pět",
            "many": "hodně",
        },
        "ordinal": {
            0: "nulakrát",
            1: "poprvé",
            2: "podruhé",
            3: "třikrát",
            4: "čtyřikrát",
            5: "pětkrát",
            "many": "hodněkrát",
        },
    },
}

PHRASES = {
    "en": {
        "SINGLE_FIRST_TIME": [
            "Wow, I see a {obj} for the first time!",
            "Looks like a new {obj} appeared.",
            "I am noticing a {obj}. Haven't seen that before.",
            "Oh! A fresh sighting: {obj}.",
            "Is that a {obj}? Interesting!",
        ],
        "SINGLE_SECOND_TIME": [
            "Ah, a {obj} again. I remember it from earlier.",
            "That's the {obj} once more. I recognize it.",
            "Hm. I’ve seen that {obj} before.",
            "The {obj} is back again!",
        ],
        "SINGLE_AGAIN": [
            "I see the {obj} again—this is the {count_word} time already!",
            "The mighty {obj}! This makes {count_word} sightings.",
            "Yes, there's a {obj}. Just like the previous {count_word} times.",
            "Oh wow, another appearance of the {obj}. {count_word} sightings and counting!",
        ],
        "SINGLE_FIRST_TIME_MULTIPLE": [
            "I see {count} {obj}s for the first time!",
            "Look at that! {count} {obj}s appeared all at once.",
            "I'm noticing {count} {obj}s. That's a first for me.",
        ],
        "SINGLE_AGAIN_MULTIPLE": [
            "I see {count} {obj}s again—this object type has been seen {count_word} times overall!",
            "{count} {obj}s! This object type is a regular sight now.",
        ],
        "MULTI_ALL_NEW": [
            "I see {objs}. All new sights!",
            "Look at that! {objs}! First time seeing them.",
            "Huh, interesting. {objs}. Haven’t seen any of those before.",
        ],
        "MULTI_ALL_KNOWN": [
            "I spot {objs} again.",
            "Here we go: {objs}, all familiar.",
            "I recognize all of these: {objs}.",
        ],
        "MULTI_MIX": [
            "I see {objs}. Some new, some familiar!",
            "There are {objs}. Interesting mix.",
            "Observing {objs}. A blend of known and unknown items.",
        ],
        "MULTI_DETAIL": [
            "I’ve seen the {obj} around here {count_word} already.",
            "That {obj}? Yeah, it shows up pretty often. {count_word} now.",
            "The {obj} is becoming a regular guest.",
        ],
        "NO_DATA": [
            "I can't see anything.",
            "There is nothing.",
            "Something is off. All darkness on my side.",
        ],
    },
    "cs": {
        "SINGLE_FIRST_TIME": [
            "{obj} je tady poprvé!",
            "Ahoj, {obj}.",
            "Vypadá to, že se objevil nový objekt: {obj}.",
            "Všiml jsem si {obj}. To jsem ještě neviděl.",
            "Ó! Nový objekt na scéně: {obj}.",
            "To je {obj}? Zajímavé!",
        ],
        "SINGLE_SECOND_TIME": [
            "Aha, znovu {obj}. Už jse známe.",
            "Znovu {obj}.",
            "Hm, {obj}? To jsem už viděl.",
            "{obj} je zpátky!",
        ],
        "SINGLE_AGAIN": [
            "Zase vidím {obj}. už {count_word}!",
            "Legendární {obj}! To už je {count_word}.",
            "Jo, zase {obj}. To už bude {count_word}.",
            "Vau, další výskyt objektu {obj}. Už {count_word} a ještě přibývá!",
        ],
        "SINGLE_FIRST_TIME_MULTIPLE": [
            "Vidím {count} {obj} poprvé!",
            "Vypadá to, že se objevilo {count} {obj} najednou.",
            "Všiml jsem si {count} {obj}. To je poprvé.",
        ],
        "SINGLE_AGAIN_MULTIPLE": [
            "Zase vidím {count} {obj}. tenhle typ už je tu {count_word}!",
            "{count} {obj}! Tohle už je tu {count_word}.",
        ],
        "MULTI_ALL_NEW": [
            "Vidím {objs}. Všechny nové!",
            "Vau, {objs}! Vidím je poprvé.",
            "Hm, zajímavé. {objs}. Nic z toho jsem ještě neviděl.",
        ],
        "MULTI_ALL_KNOWN": [
            "Vidím zase {objs}.",
            "Tak jdeme na to. {objs}. To jsou všechno známé objekty.",
            "Všechny tyto poznávám. {objs}.",
        ],
        "MULTI_MIX": [
            "Vidím {objs}. Některé nové, některé známé!",
            "Jsou tu {objs}. Zajímavý mix.",
            "Pozoruji {objs}. Směs známých i nových objektů.",
        ],
        "MULTI_DETAIL": [
            "{obj} už jsem tu viděl {count_word}.",
            "{obj}? To se tu objevuje docela často. už {count_word}.",
            "{obj} se stává pravidelným návštěvníkem.",
        ],
        "NO_DATA": ["Nic nevidím", "Nic tam není", "Asi je něco špatně. Nic nevidím."],
    },
}


def get_number_word(language, numeral_type, count):
    if count < 1:
        count = 1  # i dont want zeros really
    lang_map = NUMBER_WORDS.get(language, "en")
    numeral_map = lang_map.get(numeral_type, "ordinal")
    if count in numeral_map:
        return numeral_map[count]
    return numeral_map["many"]


def join_labels(labels, language):
    if not labels:
        return ""
    join_word = " a " if language == "cs" else " and "
    if len(labels) == 1:
        return labels[0]
    elif len(labels) == 2:
        return join_word.join(labels)
    else:
        # Use comma for all but the last two, joined by 'and'/'a'
        return ", ".join(labels[:-1]) + join_word + labels[-1]


def get_random_phrase(language, phrase_type):
    language_phrases = PHRASES.get(language, "en")
    patterns = language_phrases.get(phrase_type, "SINGLE_FIRST_TIME")
    return random.choice(patterns)
