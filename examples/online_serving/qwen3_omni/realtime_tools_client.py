"""Example client demonstrating tool calling with vLLM-Omni realtime API.

This client:
1) Connects to /v1/realtime WebSocket endpoint
2) Sends session.update with tool definitions
3) Streams audio input
4) Receives tool call requests from the model
5) Executes tools locally and sends results back
6) Receives audio response incorporating tool results

Usage:
  python realtime_tools_client.py \\
      --url ws://localhost:8091/v1/realtime \\
      --model Qwen/Qwen3-Omni-30B-A3B-Instruct \\
      --input-wav input_16k_mono.wav \\
      --output-wav tool_output.wav

Dependencies:
  pip install websockets
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import wave
from pathlib import Path
from datetime import datetime

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    raise SystemExit(1)


def _read_wav_pcm16(path: Path) -> bytes:
    """Read WAV file and validate it's mono 16-bit PCM 16kHz."""
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        comptype = wf.getcomptype()

        if nchannels != 1:
            raise ValueError(f"Input WAV must be mono (got {nchannels} channels).")
        if sampwidth != 2:
            raise ValueError(f"Input WAV must be 16-bit PCM (got sample width={sampwidth}).")
        if framerate != 16000:
            raise ValueError(f"Input WAV must be 16kHz (got {framerate} Hz).")
        if comptype != "NONE":
            raise ValueError(f"Input WAV must be uncompressed (got {comptype}).")

        return wf.readframes(wf.getnframes())


def _write_wav_pcm16(path: Path, data: bytes, sample_rate: int = 24000):
    """Write PCM16 data to WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(data)


# Example tool implementations
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city (mock implementation)."""
    # In a real implementation, this would call a weather API
    weather_data = {
        "Paris": {"temp": 18, "condition": "Partly cloudy"},
        "London": {"temp": 15, "condition": "Rainy"},
        "New York": {"temp": 22, "condition": "Sunny"},
        "Tokyo": {"temp": 25, "condition": "Clear"},
    }

    city_data = weather_data.get(city, {"temp": 20, "condition": "Unknown"})
    temp = city_data["temp"]

    if unit == "fahrenheit":
        temp = (temp * 9/5) + 32

    return json.dumps({
        "city": city,
        "temperature": temp,
        "unit": unit,
        "condition": city_data["condition"]
    })


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression (mock implementation)."""
    try:
        # Safe evaluation for simple math
        result = eval(expression, {"__builtins__": {}}, {})
        return json.dumps({"result": result, "expression": expression})
    except Exception as e:
        return json.dumps({"error": str(e), "expression": expression})


AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "calculate": calculate,
}


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specified city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name (e.g., 'Paris', 'London')"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


async def run_client(url: str, model: str, input_wav: Path, output_wav: Path):
    """Main client logic."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to {url}")

    audio_responses = []
    sample_rate = 24000

    async with websockets.connect(url) as ws:
        # 1. Receive session.created
        msg = await ws.recv()
        event = json.loads(msg)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Received: {event.get('type')}")

        # 2. Send session.update with model and tools
        session_update = {
            "type": "session.update",
            "model": model,
            "session": {
                "tools": TOOL_DEFINITIONS,
                "instructions": "You are a helpful assistant with access to tools. When the user asks about weather or calculations, use the appropriate tool."
            }
        }
        await ws.send(json.dumps(session_update))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent session.update with {len(TOOL_DEFINITIONS)} tools")

        # 3. Read and send audio input
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Reading audio from {input_wav}")
        pcm_data = _read_wav_pcm16(input_wav)

        # Send in chunks
        chunk_size = 4096
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i + chunk_size]
            b64_chunk = base64.b64encode(chunk).decode("utf-8")
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64_chunk
            }))

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent {len(pcm_data)} bytes of audio")

        # 4. Commit audio
        await ws.send(json.dumps({
            "type": "input_audio_buffer.commit",
            "final": False
        }))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Committed audio buffer")

        # 5. Listen for responses
        response_done = False
        pending_tool_calls = {}

        while not response_done:
            msg = await ws.recv()
            event = json.loads(msg)
            event_type = event.get("type")

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Received: {event_type}")

            if event_type == "response.audio.delta":
                # Collect audio chunks
                audio_b64 = event.get("audio", "")
                sample_rate = event.get("sample_rate_hz", 24000)
                audio_bytes = base64.b64decode(audio_b64)
                audio_responses.append(audio_bytes)
                print(f"  -> Audio chunk: {len(audio_bytes)} bytes")

            elif event_type == "response.audio.done":
                response_done = True
                print(f"  -> Audio response complete")

            elif event_type == "response.text.delta":
                # Text streaming
                delta = event.get("delta", "")
                print(f"  -> Text: {delta}")

            elif event_type == "response.text.done":
                text = event.get("text", "")
                print(f"  -> Full text: {text}")

            elif event_type == "response.function_call_arguments.delta":
                # Tool call in progress
                call_id = event.get("call_id")
                name = event.get("name")
                delta = event.get("delta", "")

                if call_id not in pending_tool_calls:
                    pending_tool_calls[call_id] = {
                        "id": call_id,
                        "name": name,
                        "arguments": ""
                    }

                pending_tool_calls[call_id]["arguments"] += delta
                print(f"  -> Tool call delta: {name} - {delta}")

            elif event_type == "response.function_call_arguments.done":
                # Tool call complete - execute it
                call_id = event.get("call_id")
                name = event.get("name")
                arguments_json = event.get("arguments", "{}")

                print(f"  -> Tool call complete: {name}({arguments_json})")

                # Parse arguments
                try:
                    arguments = json.loads(arguments_json)
                except json.JSONDecodeError:
                    arguments = {}

                # Execute tool
                if name in AVAILABLE_TOOLS:
                    print(f"  -> Executing tool: {name}")
                    tool_func = AVAILABLE_TOOLS[name]
                    result = tool_func(**arguments)
                    print(f"  -> Tool result: {result}")

                    # Send result back
                    tool_response = {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": result
                        }
                    }
                    await ws.send(json.dumps(tool_response))
                    print(f"  -> Sent tool result to server")

                    # Reset response_done to wait for follow-up
                    response_done = False
                else:
                    print(f"  -> ERROR: Unknown tool: {name}")

            elif event_type == "error":
                error_msg = event.get("error", {})
                print(f"  -> ERROR: {error_msg}")
                response_done = True

    # 6. Save audio output
    if audio_responses:
        combined_audio = b"".join(audio_responses)
        _write_wav_pcm16(output_wav, combined_audio, sample_rate)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saved audio to {output_wav}")
    else:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] No audio received")


def main():
    parser = argparse.ArgumentParser(description="Realtime tool calling client")
    parser.add_argument("--url", default="ws://localhost:8091/v1/realtime", help="WebSocket URL")
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model name")
    parser.add_argument("--input-wav", type=Path, required=True, help="Input WAV file (mono, 16-bit, 16kHz)")
    parser.add_argument("--output-wav", type=Path, default="tool_output.wav", help="Output WAV file")

    args = parser.parse_args()

    asyncio.run(run_client(args.url, args.model, args.input_wav, args.output_wav))


if __name__ == "__main__":
    main()
