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

The server then runs a follow-up audio generation pass that speaks the result
to the user.

## Multi-turn flow

1. Client sends `session.update` with tools and instructions.
2. Client streams user audio via `input_audio_buffer.append` and commits.
3. Server runs a text-only pass and either:
   - emits `response.function_call_arguments.delta`/`.done` (tool path), or
   - falls through to direct audio generation (no tool path).
4. On tool path, the client executes the tool and replies with
   `conversation.item.create`.
5. Server runs an audio pass that speaks the tool result.
6. Conversation history (user audio, tool calls, tool results) is retained
   for follow-up turns within the same WebSocket session.

## Running the example

Start the server with a Qwen3-Omni model:

```bash
python -m vllm_omni.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --served-model-name Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --port 8091
```

Run the example client (mock weather + calculator tools):

```bash
python examples/online_serving/qwen3_omni/realtime_tools_client.py \
    --url ws://localhost:8091/v1/realtime \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --input-wav ask_weather.wav \
    --output-wav response.wav
```

The input WAV must be mono 16-bit PCM at 16 kHz.

## Limitations

- `/v1/realtime` is not supported when `async_chunk: true` is set on the
  server. Use a stage configuration with `async_chunk: false` (the default
  for non-streaming-pipeline configs).
- Conversation history grows with each turn within a session; very long
  sessions may exceed the model's context window.
