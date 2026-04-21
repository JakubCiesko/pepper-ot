# Dialog And Speech

The robot client exposes two human interaction layers:

- QiChat grammar under `client/app/pepper-grounded-client`.
- Speech output through `SpeechAdapter` and `speech_policy.py`.

The grammar recognizes user intents and calls Python service methods. The Python code then controls server requests, speech, memory, and tablet rendering.

## QiChat Files

| File | Purpose |
|---|---|
| `client/app/pepper-grounded-client/pepper-grounded-client.dlg` | Multilanguage dialog mapping file. |
| `client/app/pepper-grounded-client/pepper-grounded-client_enu.top` | English QiChat topic. |
| `client/app/pepper-grounded-client/pepper-grounded-client_czc.top` | Czech QiChat topic. |

The `.dlg` file maps:

- English `enu` to `pepper-grounded-client_enu.top`.
- Czech `czc` to `pepper-grounded-client_czc.top`.

The package manifest declares these topics as dialog content.

## Dynamic Concepts

Both topics declare these dynamic concepts:

- `memory_objects`.
- `memory_attributes`.
- `memory_relations`.
- `memory_cached_questions`.

These are populated at runtime by `DialogAdapter.refresh_memory_concepts`, not statically in the `.top` files.

The values come from:

- `SessionStore.remembered_labels`.
- `SessionStore.remembered_attributes`.
- `SessionStore.remembered_relations`.
- `SessionStore.cached_questions`.

The source of those values is the server memory summary and Q/A pool.

## English Topic Structure

File: `client/app/pepper-grounded-client/pepper-grounded-client_enu.top`

Major sections:

### System Explanation

Concepts:

- `question_prefix`.
- `vision_action`.
- `explain_system`.

Main rule:

- `u:(~explain_system)` explains camera/server/object/relationship/memory behavior with gestures and nested follow-up rules.

Nested follow-up labels include:

- `objects`.
- `speed`.
- `scan`.
- `memory`.
- `exit`.

### Quick Look

Concepts:

- `quick_look_action`.
- `quick_look`.

Main rule calls:

```qichat
^pCall(PepperGroundedClient.look("en"))
```

This starts the quick caption/detect workflow.

### Panorama Scan

Concepts:

- `panorama_scan_action`.
- `panorama_scan`.

Main rule calls:

```qichat
^pCall(PepperGroundedClient.scan("en"))
```

This starts the full scan workflow.

### Memory Display

Concept:

- `show_memory_graph`.

Main rule calls:

```qichat
^pCall(PepperGroundedClient.showMemory("en"))
```

Nested refresh rule calls:

```qichat
^pCall(PepperGroundedClient.refreshMemoryConcepts("en"))
```

### Reset Memory

Concept:

- `reset_memory_only`.

The rule asks for confirmation. Confirmation calls:

```qichat
^pCall(PepperGroundedClient.resetMemory("en"))
```

### Reset Conversation

Concept:

- `reset_conversation_only`.

Rule calls:

```qichat
^pCall(PepperGroundedClient.resetConversation())
```

### Reset All

Concept:

- `reset_all`.

The rule asks for confirmation and then calls both memory reset and conversation reset.

### Refresh Memory Concepts

Concept:

- `refresh_memory_concepts`.

Rule calls:

```qichat
^pCall(PepperGroundedClient.refreshMemoryConcepts("en"))
```

### Object Chat

Concepts:

- `object_query_lead`.
- `object_query_prefix`.

Main rule:

```qichat
u:({~object_query_prefix} ~object_query_lead _~memory_objects {please})
```

It captures the dynamic concept value in `$1` and calls:

```qichat
^pCall(PepperGroundedClient.askAboutObject("en", $1))
```

Nested follow-ups support:

- “tell me more”.
- “what about <other memory object>”.
- Exiting the object focus.

### Listing Memory Concepts

Rules call:

- `listObjects("en")`.
- `listAttributes("en")`.
- `listRelations("en")`.

These speak sampled values from `SessionStore`.

### General Memory Questions

The topic contains rule families for:

- Counts.
- Presence.
- Position.
- Scene situation.
- Color.
- Attribute queries.
- Relation/attribute queries.

These construct a `$query` string and call:

```qichat
^pCall(PepperGroundedClient.ask("en", $query))
```

### Cached Q/A

Concept:

- `ask_suggested_questions`.

This calls `showMemory` and `listCachedQuestions`.

Direct dynamic question rule:

```qichat
u:(_~memory_cached_questions)
    ^pCall(PepperGroundedClient.answerCachedQuestion("en", $1))
```

This is the fastest voice path for pregenerated Q/A.

### Catch-All Explicit Ask

The topic has a final explicit ask pattern such as “Tell me ...” that calls `ask("en", $1)`.

## Czech Topic Structure

File: `client/app/pepper-grounded-client/pepper-grounded-client_czc.top`

The Czech topic mirrors the English topic but is currently kept as one full file rather than split by feature.

It includes the same feature groups:

- System explanation.
- Quick look.
- Panorama scan.
- Memory display.
- Reset memory.
- Reset conversation.
- Reset all.
- Refresh memory concepts.
- Object chat with `_~memory_objects`.
- Listing objects/attributes/relations.
- Counts, presence, position, situation, color, attribute, and relation questions.
- Suggested question display.
- Direct cached question answer using `_~memory_cached_questions`.
- Explicit general ask fallback.

The Czech rules call the same Python methods but pass `"cs"`.

## DialogAdapter

File: `client/app/scripts/pepper_client/interaction/dialog_adapter.py`

`DialogAdapter` wraps `ALDialog`.

### Language Resolution

`resolve_dialog_language(lang_code=None)` returns:

- `enu` for runtime English.
- `czc` for runtime Czech.

It uses `speech_policy.resolve_language_state` and current TTS language if dialog language mode is `auto`.

### Setting Dynamic Concepts

`set_dynamic_concept(name, values, language=None)`:

1. Resolves dialog language.
2. Cleans values.
3. Clears the concept if values are empty.
4. Tries multiple `ALDialog.setConcept` argument shapes because NAOqi bindings differ.

Attempted shapes:

- `(name, language, cleaned)`.
- `(name, language, [[value], ...])`.
- `(name, cleaned, language)`.
- `(name, [[value], ...], language)`.

This defensive code is intentional. Do not simplify it unless you verify exact robot binding behavior.

### Refreshing Memory Concepts

`refresh_memory_concepts(labels, attributes, relations, cached_questions=None)`:

1. Checks `dialog.enable_dynamic_memory_concepts`.
2. Cleans and caps each list based on config.
3. Updates four dynamic concepts.
4. Returns true if any concept update succeeded.

## Speech Policy

File: `client/app/scripts/pepper_client/interaction/speech_policy.py`

This module centralizes language and localized phrase selection.

### Language Functions

| Function | Purpose |
|---|---|
| `language_code(value, default="en")` | Normalizes values to `en`, `cs`, or `auto`. |
| `normalize_dialog_language(value, default="auto")` | Normalizes config language to `auto`, `czech`, or `english`. |
| `tts_language_to_runtime(tts_language)` | Maps TTS language names to `en` or `cs`. |
| `current_tts_runtime_language(tts)` | Reads current TTS language. |
| `resolve_language_state(config, requested=None, tts=None)` | Returns `(mode, runtime_lang)`. |
| `dialog_language_for_runtime(language)` | Maps runtime language to `enu` or `czc`. |
| `server_language_for_runtime(language)` | Maps runtime language to `english` or `czech`. |
| `tts_language_for_mode(mode)` | Maps config mode to `English`, `Czech`, or `None`. |

### Phrase Selection

`acknowledgement(kind, lang_code)` and `generic_message(kind, lang_code)` pick random localized strings from `_ACK` and `_GENERIC`.

Turn kinds currently using acknowledgements include:

- `look`.
- `scan`.
- `ask`.
- `reset`.

If you add a new turn kind and want acknowledgement speech, add entries here.

## SpeechAdapter

File: `client/app/scripts/pepper_client/interaction/speech_adapter.py`

`SpeechAdapter` wraps:

- `ALTextToSpeech`.
- `ALAnimatedSpeech`.

### `say(text, lang_code=None)`

Steps:

1. Cleans text with `text_utils.clean_text`.
2. Raises `SpeechError` if neither TTS service is available.
3. Acquires a lock so speech calls do not overlap at adapter level.
4. Applies language if requested/configured mode is not auto.
5. Converts unicode text to UTF-8 bytes for Python 2/NAOqi compatibility.
6. Tries `ALAnimatedSpeech.say` first.
7. Falls back to `ALTextToSpeech.say` if animated speech fails.

### `stop()`

Attempts `stopAll` on TTS and animated speech services if available.

### `_apply_language(lang_code)`

Uses `speech_policy.resolve_language_state`. If mode is:

- `auto`: do not change TTS language.
- `english`: set TTS language to `English` if available.
- `czech`: set TTS language to `Czech` if available.

## Error Policy

File: `client/app/scripts/pepper_client/utils/error_policy.py`

Defines client-specific exception classes and fallback messages.

Exceptions:

- `PepperClientError`.
- `BusyError`.
- `CameraCaptureError`.
- `ServerUnavailableError`.
- `ServerTimeoutError`.
- `MalformedResponseError`.
- `SpeechError`.
- `ConfigUpdateError`.

`fallback_message(kind, lang_code)` supports:

- `busy`.
- `camera`.
- `server_unavailable`.
- `server_timeout`.
- `malformed`.
- `unexpected`.

## Text Utilities

File: `client/app/scripts/pepper_client/utils/text.py`

The client still targets Python 2, so text handling is deliberately defensive.

Functions:

| Function | Behavior |
|---|---|
| `clean_text(text, max_chars=None)` | Converts to unicode, removes newlines, collapses whitespace, strips, truncates, returns text. |
| `sanitize_query(query, max_chars)` | Alias around `clean_text` with max length. |
| `clean_text_unicode(text, max_chars=None)` | Unicode-literal variant used for Czech-safe speech/listing. |

Important compatibility note: this module uses `unicode` and `ur"..."`, so it is Python 2 source. Running it under Python 3 will fail unless changed.

## Where To Add Grammar

When adding a new spoken feature:

1. Add English grammar to `pepper-grounded-client_enu.top`.
2. Add Czech grammar to `pepper-grounded-client_czc.top` if the feature should be bilingual.
3. Prefer dynamic concepts for object/relation/attribute-specific grammar.
4. Keep QiChat rules thin and call a bound Python method.
5. Put long-running logic in `TurnManager`.
6. Update dynamic concept refresh if the feature needs new concept types.
7. Update this doc and [`runtime-service-and-turns.md`](runtime-service-and-turns.md).
