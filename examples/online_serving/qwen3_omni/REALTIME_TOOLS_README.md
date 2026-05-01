# Qwen3-Omni Realtime API with Tool Calling

This example demonstrates the OpenAI-compatible `/v1/realtime` WebSocket API
with tool calling support for Qwen3-Omni.

The server accepts streamed PCM audio, detects tool calls in the model's
response, emits OpenAI-style function-call events, accepts tool results from
the client, and produces speech audio incorporating the result.

## Supported events

**Client → Server:**

| Type | Purpose |
| --- | --- |
| `session.update` | Configure model, instructions, and tool definitions |
| `input_audio_buffer.append` | Send a chunk of base64-encoded 16 kHz mono PCM16 audio |
| `input_audio_buffer.commit` | Trigger generation on the buffered audio |
| `conversation.item.create` | Send a `function_call_output` tool result back to the model |

**Server → Client:**

| Type | Purpose |
| --- | --- |
| `session.created` | Connection established |
| `response.text.delta` / `response.text.done` | Incremental and final transcript text |
| `response.function_call_arguments.delta` / `.done` | Tool call name + JSON arguments |
| `response.audio.delta` | Base64-encoded 24 kHz PCM16 chunk |
| `response.audio.done` | Audio response complete |
| `error` | Error event (for unsupported configurations or runtime errors) |

## `session.update` payload

`session.update` carries the served model name and the session config
(instructions and tool definitions):

```json
{
  "type": "session.update",
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "session": {
    "instructions": "You are a helpful voice assistant. Use tools when appropriate.",
    "tools": [ /* see Tool definition format below */ ]
  }
}
```

## Tool definition format

Tool definitions follow OpenAI's JSON schema format and are passed in
`session.update` under `session.tools`:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
]
```

## Tool call wire format

Qwen3-Omni emits tool calls in its native XML-wrapped JSON format. The server
parses this format and translates it to OpenAI-style streaming events for the
client. From the client's perspective, you only see standard OpenAI events;
the underlying model output looks like:

```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
</tool_call>
```

## Tool result format

After receiving a `response.function_call_arguments.done` event, the client
executes the tool and sends back the result with the matching `call_id`:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "function_call_output",
    "call_id": "call_abc123",
    "output": "{\"temperature\": 18, \"condition\": \"Sunny\"}"
  }
}
```

The tool result is appended to the session's conversation history. The server
then runs a follow-up audio generation pass with the full conversation
context (instructions + history including the new tool result) and the
model decides how to respond — paraphrase the result, ask a clarifying
question, combine multiple tool outputs, etc. — based on its session
instructions and what the conversation calls for.

## Multi-turn flow

1. Client sends `session.update` with tools and instructions.
2. Client streams user audio via `input_audio_buffer.append` and commits.
3. Server runs a text-only pass and either:
   - emits `response.function_call_arguments.delta`/`.done` (tool path), or
   - falls through to direct audio generation (no tool path).
4. On tool path, the client executes the tool and replies with
   `conversation.item.create`.
5. Server runs an audio pass with the conversation context (session
   instructions + history including the tool result). The model decides
   how to respond.
6. Conversation history (user audio, tool calls, tool results) is retained
   for follow-up turns within the same WebSocket session.

## Running the example

`/v1/realtime` requires a deployment config with `async_chunk: false`. The
default Qwen3-Omni config (`vllm_omni/deploy/qwen3_omni_moe.yaml`) has
`async_chunk: true` and is therefore not usable here. A ready-to-use config
tuned for 2× A100 80 GB is bundled alongside this README at
`qwen3_omni_moe_realtime.yaml` — see the header of that file for the GPU
memory math if you need to retune for different hardware.

Start the server:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --deploy-config examples/online_serving/qwen3_omni/qwen3_omni_moe_realtime.yaml
```

Run the example client (mock weather + calculator tools):

```bash
python examples/online_serving/qwen3_omni/realtime_tools_client.py \
    --url ws://localhost:8091/v1/realtime \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --input-wav ask_weather.wav \
    --output-wav response.wav
```

`--input-wav` accepts multiple files to run sequential turns over a single
WebSocket session, demonstrating that conversation context is retained
across turns:

```bash
python examples/online_serving/qwen3_omni/realtime_tools_client.py \
    --input-wav greeting.wav weather_paris.wav weather_london.wav \
    --output-wav response.wav   # writes response_turn1.wav, _turn2.wav, _turn3.wav
```

Input WAVs must be mono 16-bit PCM at 16 kHz.

## Limitations

- `/v1/realtime` requires `async_chunk: false` in the deployment config
  (see "Running the example" above).
- Conversation history grows with each turn within a session; very long
  sessions may exceed the model's context window.
- The audio-generation pass does not detect or abort on `<tool_call>`
  output. As a result, the model cannot chain a second tool call in the
  same turn — it can only respond to the existing tool result. Chained
  tool calls within a single turn would require adding tool-call detection
  and abort handling to the post-tool audio pass.
- When tools or session instructions are configured, every turn runs a
  two-pass generation: a text-only thinker pass for tool-call detection
  followed by a separate full-stage pass for audio. The talker stage
  consumes thinker hidden states, and the engine does not currently
  retain those states across `engine.generate()` calls (nor expose an API
  to feed cached hidden states into a later stage), so the second pass
  must re-run the thinker. Eliminating it would require engine-level
  changes (hidden-state caching across stages, or a conditional-abort
  scheduler that lets the talker proceed only if no tool call is
  detected). This is a per-turn cost, not a multiplier on tool calls.
- The two passes use different prompt structures. The text-only thinker
  pass wraps the full system context (tools + instructions + conversation
  history) into a single `system` block before the user audio, because
  `buffer_realtime_audio` does not accept a structured message list. The
  audio pass uses proper Qwen3 chat-template format: a `system` block
  with tools and instructions, prior turns as `assistant`/`user` blocks,
  then the user audio block, then any current-turn tool interaction, then
  the final `assistant` block. Even at temperature 0, the regenerated
  assistant tokens in the second pass can drift slightly from what the
  client already received via `response.text.delta`. In practice the
  divergence is small, but the spoken audio is not guaranteed to match
  the streamed transcript verbatim.
- Conversation history is maintained on the assistant side only. User
  speech is not transcribed or stored across turns, so the model
  reconstructs user intent from the live audio combined with its own
  prior tool-call breadcrumbs. Background post-turn transcription is
  theoretically possible (the engine is idle while the user listens to
  the response), but requires retroactive insertion of user-turn items
  into conversation history with abort handling if the user speaks before
  transcription finishes — left as a future enhancement.
