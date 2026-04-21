# Session State And Dynamic Concepts

The robot client keeps local state only for the current client process. It does not persist scene memory or conversations. Persistent visual memory and conversation storage live on the server.

The client state layer exists to make robot interaction smooth:

- It remembers the current chat id.
- It tracks when visual context was last refreshed.
- It stores labels/relations/attributes for speech grammar dynamic concepts.
- It caches pregenerated Q/A pairs for fast spoken answers.
- It stores the last server response for status/debugging.

## SessionStore

File: `client/app/scripts/pepper_client/core/session_store.py`

`SessionStore` is a thread-safe state holder using `threading.RLock`.

Initial state keys:

| Key | Meaning |
|---|---|
| `chat_id` | Current server conversation id. |
| `last_caption` | Last caption text. |
| `last_caption_ts` | Timestamp of last caption. |
| `last_detect_ts` | Timestamp of last detect or scan detect response. |
| `last_scan_id` | Last generated scan id. |
| `last_response` | Last full server response or summary. |
| `last_query` | Last chat query sent to server. |
| `last_detect_request_id` | Detect id returned by caption endpoint if any. |
| `last_server_base_url` | Server URL currently configured. |
| `remembered_labels` | Sorted unique object labels from memory summary. |
| `remembered_attributes` | Sorted unique unary relation names from memory summary. |
| `remembered_relations` | Sorted unique binary relation names from memory summary. |
| `last_memory_summary` | Last raw memory summary object. |
| `last_memory_summary_ts` | Timestamp of last memory summary update. |
| `cached_questions` | Sorted unique pregenerated questions. |
| `cached_answers` | Mapping from exact question text to answer. |

## Reset Methods

### `reset_all()`

Clears all state to initial values. Called by constructor.

### `reset_conversation()`

Clears conversation-related fields:

- `chat_id`.
- `last_response`.
- `last_query`.
- `last_caption`.
- `last_caption_ts`.
- `last_detect_request_id`.

It does not clear remembered memory labels or cached Q/A.

### `reset_memory_state()`

Clears visual memory-related local fields:

- `remembered_labels`.
- `remembered_attributes`.
- `remembered_relations`.
- `last_memory_summary`.
- `last_memory_summary_ts`.
- `cached_questions`.
- `cached_answers`.

It does not clear `chat_id` by itself, but `TurnManager._run_reset_memory` calls both memory reset and conversation reset.

## Update Methods

### `update_after_caption(caption_response)`

Stores:

- `caption_response.caption` in `last_caption`.
- Current timestamp in `last_caption_ts`.
- `caption_response.detect_request_id` in `last_detect_request_id`.
- Full response in `last_response`.

This is used after quick look.

### `update_after_detect(detect_response, scan_id=None)`

Stores:

- Current timestamp in `last_detect_ts`.
- Scan id in `last_scan_id`.
- Full detect response in `last_response`.

This is used after visual refresh, sequential scan frame, and panorama scan.

### `update_after_chat(query, chat_response)`

Stores:

- `chat_response.chat_id` in `chat_id`.
- Query in `last_query`.
- Full response in `last_response`.

This is what lets future chat/object-chat calls continue the same server conversation.

### `set_server_base_url(value)`

Stores server URL for status/debugging.

### `update_after_memory_summary(summary)`

Consumes server memory summary and populates dynamic concept source lists.

It reads:

- `summary.labels` for labels.
- `summary.scene_graph` for attributes and relations.

Edge splitting rule:

```text
if sub and obj and sub == obj: attribute
else: relation
```

Only relation names are stored for attributes/relations dynamic concepts, not full edges.

### `update_after_pregenerated_qa(qa_response)`

Consumes Q/A response and populates:

- `cached_questions`.
- `cached_answers`.

Only pairs with non-empty question and answer are accepted.

Questions are cleaned with `clean_text_unicode`; answers are also cleaned with `clean_text_unicode`.

## Read Methods

Important getters:

- `get_cached_questions()`.
- `get_cached_answers()`.
- `get_memory_labels()`.
- `get_memory_attributes()`.
- `get_memory_relations()`.
- `get_chat_id()`.
- `snapshot()`.

`get_last_memory_summary()` exists but is marked as never used.

## Visual Refresh TTL

`needs_visual_refresh(ttl_seconds)` returns true if:

- No detect has ever run.
- Time since `last_detect_ts` exceeds configured TTL.

`TurnManager._run_ask` uses this when `behavior.auto_refresh_before_chat=true`.

## Sorting And Deduplication

`_sorted_unique(values)` cleans text, removes duplicates, sorts output, and returns a list.

This is why dynamic concept values may appear alphabetically rather than in detection order.

## Dynamic Concept Refresh Flow

The full refresh path is:

1. `TurnManager._refresh_dynamic_concepts_from_server` calls `/api/v1/memory/summary`.
2. `SessionStore.update_after_memory_summary` stores labels, attributes, and relations.
3. `_refresh_dynamic_concepts_from_summary` extracts labels and relation names from summary.
4. `DialogAdapter.refresh_memory_concepts` caps and cleans values.
5. ALDialog receives updated dynamic concepts.

This happens after:

- Client startup.
- Quick look when configured.
- Visual refresh before chat when configured.
- Scan when configured.
- Memory display.
- Memory reset when configured.

## Cached Q/A Flow

The Q/A cache is populated mainly during `showMemory`:

1. Client fetches memory summary.
2. Client calls `/api/v1/chat/pregenerate_qa`.
3. `SessionStore.update_after_pregenerated_qa` stores exact question/answer pairs.
4. `DialogAdapter.refresh_memory_concepts` inserts questions into `memory_cached_questions`.
5. Voice rule `u:(_~memory_cached_questions)` can call `answerCachedQuestion`.
6. Tablet Q/A button can call `answerCachedQuestion` through `qi.js`.

If the exact cached answer exists, the robot speaks without server chat latency. If it misses, `TurnManager._run_cached_answer` falls back to general chat.

## Tablet Payload State

`TurnManager._build_memory_page_payload(summary, qa_response, ui_language)` builds the object passed to tablet rendering.

It does not store that payload separately. It derives it from current summary and Q/A response each time memory is shown.

Payload fields:

- `ui_language`.
- `object_labels`.
- `label_counts`.
- `attributes`.
- `relationships`.
- `graph_svg`.
- `pregenerated_qa`.
- `qa_metadata` when present.

## State Ownership Boundary

The client owns temporary interaction state. The server owns visual world state.

If you need to remember objects across frames, edit server memory code, not `SessionStore`. The client should only mirror memory summaries for grammar and UI.
