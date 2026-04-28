# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extended RealtimeConnection with tool calling support for vLLM Omni."""

from __future__ import annotations

import asyncio
import base64
import copy
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

import numpy as np
import torch
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.openai.realtime.connection import (
    RealtimeConnection as VllmRealtimeConnection,
)
from vllm.entrypoints.openai.realtime.protocol import (
    InputAudioBufferCommit,
    TranscriptionDelta,
    TranscriptionDone,
)
from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.utils import coerce_param_message_types

logger = init_logger(__name__)


class RealtimeConnection(VllmRealtimeConnection):
    """Extended RealtimeConnection with tool calling support for Qwen3 Omni.

    This class extends vLLM's base RealtimeConnection to add:
    - Tool configuration via session.update
    - XML tool call detection and streaming (Qwen3 format)
    - Tool result handling via conversation.item.create
    - Multi-turn conversation context management
    - Text streaming alongside audio
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = cast(AsyncOmni, self.serving.engine_client)
        self._realtime_audio_ref: np.ndarray | None = None

        # Tool calling and conversation state
        self.tools: list[dict] | None = None
        self.instructions: str | None = None
        self.conversation_items: list[dict] = []
        self.conversation_context: str | None = None

        # Tool call parsing state
        self.in_tool_call = False
        self.tool_call_buffer = ""
        self.current_tool_calls: list[dict] = []
        self.current_function_name: str | None = None
        self.current_tool_call_id: str | None = None
        self.accumulated_text = ""
        # Cursor into accumulated_text marking the start of the next unprocessed region.
        # After a <tool_call>...</tool_call> block is parsed, we advance this past the
        # block so we don't re-detect the same tool call on every subsequent token.
        self._text_proc_cursor: int = 0

        # Set to True when a tool result arrives while the text pass is still active.
        # _run_generation checks this flag before returning and immediately schedules
        # _run_audio_from_tool_context so the result is not lost.
        self._pending_tool_context: bool = False

        # Cache user audio chunks for replay in the audio pass after tool results.
        # The talker needs audio-grounded hidden states from the thinker — a text-only
        # prompt produces garbled audio.
        self._cached_user_audio: list[np.ndarray] = []

        # Acoustic reference for the audio pass. Set once from the first turn's user
        # audio which produces clean speech through hidden_projection. Later turns may
        # use a different speaker; keeping the first-turn reference maintains voice
        # quality across turns.
        self._turn_audio_cache: list[np.ndarray] | None = None

        # Tokenizer for decoding thinker text tokens
        self.tokenizer = None
        try:
            model_config = self.serving.model_config
            self.tokenizer = cached_tokenizer_from_config(model_config)
            logger.debug("Tokenizer loaded for tool call parsing")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")

    # -------------------------------------------------------------------------
    # Event handling
    # -------------------------------------------------------------------------

    async def handle_event(self, event: dict):
        """Override to handle tool-related events."""
        event_type = event.get("type")

        if event_type == "session.update":
            session = event.get("session", {})
            self.tools = session.get("tools")
            self.instructions = session.get("instructions")
            logger.info(f"Session updated with {len(self.tools) if self.tools else 0} tools")
            await super().handle_event(event)

        elif event_type == "input_audio_buffer.commit":
            # Override commit handling: start generation AND close the audio stream
            # so buffer_realtime_audio() can flush and finish, which in turn lets
            # _add_streaming_input_request send resumable=False to the engine.
            # Without the None sentinel, the thinker stage never gets finished=True
            # and never forwards to talker→code2wav, so audio never arrives.
            commit_event = InputAudioBufferCommit(**event)
            if commit_event.final:
                self.audio_queue.put_nowait(None)
            else:
                await self.start_generation()
                self.audio_queue.put_nowait(None)

        elif event_type == "conversation.item.create":
            item = event.get("item", {})
            await self._handle_conversation_item(item)

        else:
            await super().handle_event(event)

    async def _handle_conversation_item(self, item: dict):
        """Handle conversation item creation (e.g., tool results)."""
        item_type = item.get("type")

        if item_type == "function_call_output":
            tool_result = {
                "role": "tool",
                "content": item.get("output", ""),
                "call_id": item.get("call_id"),
            }
            self.conversation_items.append(tool_result)
            logger.info(f"Received tool result for call_id: {tool_result['call_id']}")
            await self._generate_with_tool_context()

        elif item_type == "message":
            self.conversation_items.append(item)
            logger.debug(f"Added message to conversation: {item.get('role', 'unknown')}")

    async def _generate_with_tool_context(self):
        """Run audio-only generation after tool results have been received."""
        if self.generation_task is not None and not self.generation_task.done():
            # Text pass is still running — mark so _run_generation picks it up.
            logger.info(
                "Text pass still running — scheduling audio pass to start once it finishes"
            )
            self._pending_tool_context = True
            return

        logger.info("Generating audio response with tool context")
        self.generation_task = asyncio.create_task(
            self._run_audio_from_tool_context()
        )

    # -------------------------------------------------------------------------
    # Sampling params helpers
    # -------------------------------------------------------------------------

    def _make_audio_sampling_params_list(self):
        """Per-stage sampling params for realtime audio generation.

        Forces thinker (stage 0) and talker (stage 1) to temperature=0.0 and
        coerces all stages to DELTA output so code2wav batches are emitted to
        the client incrementally rather than buffered until completion.
        NOTE: a single sampling_params= to engine.generate() does NOT override
        per-stage YAML defaults — only sampling_params_list= is respected.
        """
        default_spl = getattr(self.engine, "default_sampling_params_list", None)
        if default_spl is None or len(default_spl) < 2:
            logger.warning(
                "default_sampling_params_list unavailable (got %s) — "
                "audio pass will use YAML defaults unchanged",
                default_spl,
            )
            return None
        spl = copy.deepcopy(list(default_spl))
        spl[0].temperature = 0.0
        spl[1].temperature = 0.0
        return coerce_param_message_types(spl, is_streaming=True)

    def _make_text_sampling_params_list(self):
        """Per-stage sampling params for the text-only thinker pass.

        Forces thinker (stage 0) to temperature=0.0 so tool call detection is
        fast and deterministic. Coerces all stages to DELTA output so each step
        yields only NEW tokens — the loop in _run_generation does
        `accumulated_text += _decode_tokens(token_ids)`, which would duplicate
        text quadratically if token_ids were cumulative. Cumulative bloat saved
        to conversation history then poisons subsequent turns.
        """
        default_spl = getattr(self.engine, "default_sampling_params_list", None)
        if not default_spl:
            return None
        spl = copy.deepcopy(list(default_spl))
        spl[0].temperature = 0.0
        return coerce_param_message_types(spl, is_streaming=True)

    # -------------------------------------------------------------------------
    # Generation entry point
    # -------------------------------------------------------------------------

    async def start_generation(self):
        """Start the transcription generation task with conversation context support."""
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        audio_stream = self.audio_stream_generator()
        input_stream = asyncio.Queue[list[int]]()

        # IMPORTANT: Only inject conversation_context when a text-only pass will run
        # (i.e. tools are configured OR session instructions are set).
        # Audio-only passes MUST receive context=None because buffer_realtime_audio
        # splits a non-None context into a separate initial add_request, leaving
        # audio tokens as streaming updates. That makes thinker_output.prompt_token_ids
        # contain only system tokens — _compute_talker_prompt_ids_length finds no
        # <|im_start|>user marker, returns 0, and the talker hits the decode path
        # instead of prefill, crashing with "Missing prefill_consumed_text_tokens".
        conversation_context = getattr(self, "conversation_context", None)
        if conversation_context is None and (self.tools or self.instructions):
            conversation_context = self._build_system_context()

        streaming_input_gen = self.serving.transcribe_realtime(
            audio_stream, input_stream, conversation_context
        )
        self.conversation_context = None

        self.generation_task = asyncio.create_task(
            self._run_generation(streaming_input_gen, input_stream)
        )

    # -------------------------------------------------------------------------
    # Audio utilities (from main — delta deduplication + format conversion)
    # -------------------------------------------------------------------------

    @staticmethod
    def _tensor_to_numpy(value) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "detach"):
            arr = value.detach().float().cpu().numpy()
        else:
            try:
                arr = np.asarray(value)
            except Exception:
                return None
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _numpy_audio_prefix_match(prev: np.ndarray, curr: np.ndarray) -> bool:
        n = prev.shape[0]
        if n == 0:
            return True
        if curr.shape[0] < n:
            return False
        return bool(np.allclose(curr[:n], prev, rtol=1e-3, atol=2e-4))

    def _raw_waveform_to_deltas(self, arr: np.ndarray) -> list[np.ndarray]:
        """Convert one streaming PCM f32 chunk into incremental piece(s).

        Handles both cumulative-waveform and true-delta engine output modes
        without duplicating audio on the client.
        """
        if arr.size == 0:
            return []
        ref = self._realtime_audio_ref
        if ref is None:
            self._realtime_audio_ref = arr.copy()
            return [arr]
        if self._numpy_audio_prefix_match(ref, arr):
            delta = arr[ref.shape[0]:]
            self._realtime_audio_ref = arr.copy()
            return [delta] if delta.size > 0 else []
        self._realtime_audio_ref = np.concatenate([ref, arr])
        return [arr]

    def _extract_audio_chunks(self, output) -> tuple[list[np.ndarray], int]:
        mm = getattr(output, "multimodal_output", None)
        if not isinstance(mm, dict):
            return [], 24000

        sr = mm.get("sr") or mm.get("sample_rate") or mm.get("audio_sample_rate") or 24000
        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return [], int(sr)

        raw_audio = mm.get(key)
        chunks: list[np.ndarray] = []
        if isinstance(raw_audio, (list, tuple)):
            if len(raw_audio) > 0:
                arr = self._tensor_to_numpy(raw_audio[-1])
                if arr is not None and arr.size > 0:
                    chunks.extend(self._raw_waveform_to_deltas(arr))
        else:
            arr = self._tensor_to_numpy(raw_audio)
            if arr is not None and arr.size > 0:
                chunks.extend(self._raw_waveform_to_deltas(arr))
        return chunks, int(sr)

    @staticmethod
    def _pcm16_b64(audio_f32: np.ndarray) -> str:
        clipped = np.clip(audio_f32, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("utf-8")

    # Maximum raw PCM bytes per WebSocket message for response.audio.delta.
    # Base64 encoding inflates by ~4/3, so 200 KB raw → ~267 KB on the wire.
    _AUDIO_DELTA_CHUNK_BYTES: int = 200 * 1024

    async def _send_audio_delta(self, chunk_f32: np.ndarray, sample_rate: int) -> None:
        """Send a f32 PCM chunk as one or more response.audio.delta messages.

        Converts to int16 and splits into _AUDIO_DELTA_CHUNK_BYTES pieces so
        no single WebSocket frame exceeds client size limits.
        """
        raw = (np.clip(chunk_f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        size = self._AUDIO_DELTA_CHUNK_BYTES
        for i in range(0, max(len(raw), 1), size):
            piece = raw[i:i + size]
            if not piece:
                break
            await self.send_json(
                {
                    "type": "response.audio.delta",
                    "audio": base64.b64encode(piece).decode("utf-8"),
                    "sample_rate_hz": sample_rate,
                }
            )

    # -------------------------------------------------------------------------
    # Generation loops
    # -------------------------------------------------------------------------

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        """Override generation to add text streaming and tool call detection."""
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        sent_audio = False
        done_sent = False
        self._realtime_audio_ref = None

        # Reset state for new generation
        self.accumulated_text = ""
        self.current_tool_calls = []
        self._text_proc_cursor = 0

        try:
            if self.tools or self.instructions:
                # --- Text-only pass when tools are configured or instructions are set ---
                # Run the thinker stage only (output_modalities=["text"]).
                # If the model makes a tool call, we emit the event and stop;
                # the client sends back a conversation.item.create with
                # function_call_output, which triggers _run_audio_from_tool_context.
                # If no tool calls, fall through to _run_audio_from_tool_context
                # which rebuilds the prompt from _cached_user_audio.
                text_request_id = request_id + "-txt"
                result_gen = self.engine.generate(
                    prompt=streaming_input_gen,
                    sampling_params_list=self._make_text_sampling_params_list(),
                    request_id=text_request_id,
                    output_modalities=["text"],
                )
                async for output in result_gen:
                    if output.outputs:
                        token_ids = list(output.outputs[0].token_ids)
                        if token_ids:
                            text_delta = self._decode_tokens(token_ids)
                            if text_delta:
                                await self._process_text_delta(text_delta)
                    # Abort as soon as a complete tool call is detected — don't
                    # let the thinker keep generating hallucinated post-tool text.
                    if self.current_tool_calls:
                        await self.engine.abort(text_request_id)
                        break
                    if not self._is_connected:
                        break

                self.conversation_items.append(
                    {
                        "role": "assistant",
                        "content": self.accumulated_text or None,
                        "tool_calls": self.current_tool_calls or None,
                    }
                )

                if self.current_tool_calls:
                    logger.info(
                        "Emitted %d tool call(s); waiting for client tool responses",
                        len(self.current_tool_calls),
                    )
                    if self.accumulated_text:
                        await self.send_json({"type": "response.text.done", "text": self.accumulated_text})
                    while not self.audio_queue.empty():
                        self.audio_queue.get_nowait()
                    if self._pending_tool_context:
                        self._pending_tool_context = False
                        logger.info("Tool result was already received — starting audio pass now")
                        self.generation_task = asyncio.create_task(
                            self._run_audio_from_tool_context()
                        )
                    return

                # No tool calls — start audio pass for direct response.
                logger.info("No tool call detected — starting direct audio pass")
                self.generation_task = asyncio.create_task(
                    self._run_audio_from_tool_context()
                )
                return

            else:
                # No tools and no instructions — single audio pass (fast path)
                result_gen = self.engine.generate(
                    prompt=streaming_input_gen,
                    request_id=request_id,
                    output_modalities=["audio"],
                    sampling_params_list=self._make_audio_sampling_params_list(),
                )
                full_text = ""
                prompt_token_ids_len = 0
                completion_tokens_len = 0
                async for output in result_gen:
                    if output.outputs and len(output.outputs) > 0:
                        first_output = output.outputs[0]
                        new_token_ids = list(first_output.token_ids)
                        if not prompt_token_ids_len and output.prompt_token_ids:
                            prompt_token_ids_len = len(output.prompt_token_ids)
                        if new_token_ids:
                            input_stream.put_nowait(new_token_ids)
                        delta_text = first_output.text or ""
                        full_text += delta_text
                        if delta_text:
                            await self.send(TranscriptionDelta(delta=delta_text))
                        completion_tokens_len += len(new_token_ids)
                    audio_chunks, sample_rate = self._extract_audio_chunks(output)
                    for chunk in audio_chunks:
                        sent_audio = True
                        await self._send_audio_delta(chunk, sample_rate)
                    if not self._is_connected:
                        break

                self.conversation_items.append(
                    {
                        "role": "assistant",
                        "content": full_text or None,
                        "tool_calls": None,
                    }
                )
                usage = UsageInfo(
                    prompt_tokens=prompt_token_ids_len,
                    completion_tokens=completion_tokens_len,
                    total_tokens=prompt_token_ids_len + completion_tokens_len,
                )
                await self.send(TranscriptionDone(text=full_text, usage=usage))

            if sent_audio:
                await self.send_json({"type": "response.audio.done", "has_audio": True})
                done_sent = True

            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            if self._is_connected and not done_sent and sent_audio:
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": True})
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    async def _run_audio_from_tool_context(self) -> None:
        """Generate speech after receiving tool results (or for direct responses).

        Prompt structure (NO system section):

            <|im_start|>user
            <|audio_start|><|audio_pad|><|audio_end|>
            <|im_end|>
            <|im_start|>user
            {instructions + tool result — text-only, SKIPPED by talker}
            <|im_end|>
            <|im_start|>assistant

        Why this works:
        - Audio user section is at position [0:15] — identical to Phase 1.
          The thinker has ZERO preceding tokens before the audio, so its
          layer-24 hidden states at audio positions are purely acoustic.
          hidden_projection maps these clean acoustic states → clear speech.
        - The text instruction user section is text-only (no audio tokens),
          so our fix in _thinker_to_talker_prefill SKIPS it. The talker
          never sees the contaminating text embeddings.
        - Causal attention means the audio tokens (positions 0-14) cannot
          attend forward to the instruction (positions 15+), keeping the
          audio hidden states acoustically pure.
        - The thinker's assistant tokens CAN attend to both the audio AND
          the instruction, generating a response about the tool result.
        """
        sent_audio = False
        done_sent = False
        self._realtime_audio_ref = None
        try:
            audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"

            # Collect tool results for the CURRENT turn's tool calls only.
            current_call_ids = {tc.get("id") for tc in self.current_tool_calls}
            tool_result_parts: list[str] = []
            for item in self.conversation_items:
                if item.get("role") == "tool":
                    call_id = item.get("call_id")
                    if call_id not in current_call_ids:
                        continue
                    result_content = item.get("content", "")
                    tool_name = next(
                        (tc["name"] for tc in self.current_tool_calls if tc.get("id") == call_id),
                        "tool",
                    )
                    tool_result_parts.append(f"{tool_name} returned: {result_content}")

            tool_result_summary = "\n".join(tool_result_parts)

            # Two cases:
            # (A) Tool-result turn: use ONLY the "Speak this" directive — no session
            #     instructions, no history. Session instructions may contain explicit
            #     tool-calling directives that bleed into the audio pass and push the
            #     thinker toward tool-call token patterns rather than speech bootstrap.
            # (B) Direct-response turn (no tool result): include session instructions
            #     plus any prior conversation history for follow-up questions.
            if tool_result_summary:
                instruction_text = f"Speak this information to the user: {tool_result_summary}"
            else:
                instruction_parts: list[str] = []
                if self.instructions:
                    instruction_parts.append(self.instructions)
                if self.conversation_items:
                    history = self._build_conversation_context()
                    if history:
                        instruction_parts.append(f"Conversation history:\n{history}")
                instruction_text = "\n".join(instruction_parts)

            parts: list[str] = []
            parts.append(f"<|im_start|>user\n{audio_placeholder}<|im_end|>")
            parts.append(f"<|im_start|>user\n{instruction_text}<|im_end|>")
            parts.append("<|im_start|>assistant")
            full_prompt = "\n".join(parts) + "\n"

            # Use current-turn audio for acoustic conditioning so the talker's
            # hidden_projection receives full-length in-distribution features.
            # Fall back to the first-turn cache only if the current buffer is empty.
            ref_audio = self._cached_user_audio if self._cached_user_audio else self._turn_audio_cache
            audio_array = np.concatenate(ref_audio) if ref_audio else np.zeros(8000, dtype=np.float32)

            prompt = {
                "prompt": full_prompt,
                "multi_modal_data": {
                    "audio": (audio_array, 16000),
                },
            }

            request_id = f"rt-{self.connection_id}-{uuid4()}-aud"
            result_gen = self.engine.generate(
                prompt=prompt,
                request_id=request_id,
                output_modalities=["audio"],
                sampling_params_list=self._make_audio_sampling_params_list(),
            )

            async for output in result_gen:
                audio_chunks, sample_rate = self._extract_audio_chunks(output)
                for chunk in audio_chunks:
                    sent_audio = True
                    await self._send_audio_delta(chunk, sample_rate)
                if not self._is_connected:
                    break

            if sent_audio:
                await self.send_json({"type": "response.audio.done", "has_audio": True})
                done_sent = True

        except Exception as exc:
            logger.exception("Error in audio-from-tool-context pass: %s", exc)
            await self.send_error(str(exc), "processing_error")
        finally:
            if not done_sent and sent_audio:
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": True})
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # Conversation helpers
    # -------------------------------------------------------------------------

    def _build_conversation_context(self) -> str:
        """Build a text representation of conversation history."""
        context_parts = []
        for item in self.conversation_items:
            role = item.get("role", "")
            if role == "user":
                content = item.get("content", "[User spoke]")
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                content = item.get("content", "")
                tool_calls = item.get("tool_calls", [])
                if content:
                    clean = self._clean_text_delta(content)
                    if clean:
                        context_parts.append(f"Assistant: {clean}")
                if tool_calls:
                    for call in tool_calls:
                        args_str = json.dumps(call.get("arguments", {}))
                        context_parts.append(f"Assistant called: {call.get('name')}({args_str})")
            elif role == "tool":
                call_id = item.get("call_id", "")
                content = item.get("content", "")
                context_parts.append(f"Tool result [{call_id}]: {content}")
        return "\n".join(context_parts)

    def _format_tools_for_prompt(self) -> str:
        """Format tools using Qwen3's exact chat-template convention."""
        if not self.tools:
            return ""
        tool_lines = "\n".join(json.dumps(t) for t in self.tools)
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tool_lines}\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{\"name\": <function-name>, \"arguments\": <args-json-object>}\n'
            "</tool_call>"
        )

    def _build_system_context(self) -> str | None:
        """Build the system context string from instructions + tools + conversation history."""
        parts: list[str] = []
        if self.tools:
            parts.append(self._format_tools_for_prompt())
        if self.instructions:
            parts.append(self.instructions)
        if self.conversation_items:
            history = self._build_conversation_context()
            if history:
                parts.append(history)
        return "\n\n".join(parts) if parts else None

    def audio_stream_generator(self):
        """Override to cache audio chunks for replay in tool-context audio pass."""
        async def _gen():
            self._cached_user_audio = []
            while True:
                audio_chunk = await self.audio_queue.get()
                if audio_chunk is None:
                    break
                self._cached_user_audio.append(audio_chunk)
                yield audio_chunk
            # Persist the first turn's audio as the acoustic reference.
            if self._turn_audio_cache is None and self._cached_user_audio:
                self._turn_audio_cache = list(self._cached_user_audio)
        return _gen()

    # -------------------------------------------------------------------------
    # Text processing helpers
    # -------------------------------------------------------------------------

    def _decode_tokens(self, token_ids: list[int]) -> str:
        if not self.tokenizer or not token_ids:
            return ""
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception as e:
            logger.warning(f"Failed to decode tokens: {e}")
            return ""

    def _parse_tool_call(self, tool_call_block: str) -> dict | None:
        """Parse a complete Qwen3 <tool_call>...</tool_call> block."""
        try:
            inner = tool_call_block.split("<tool_call>", 1)[1].split("</tool_call>")[0].strip()
            parsed = json.loads(inner)
            name = parsed.get("name")
            arguments = parsed.get("arguments", {})
            if not name:
                logger.debug("Tool call missing 'name' field: %s", inner[:100])
                return None
            return {"name": name, "arguments": arguments}
        except Exception as exc:
            logger.warning("Failed to parse tool call block: %s", exc)
            return None

    async def _process_text_delta(self, text_delta: str):
        """Process text delta for tool call detection and streaming.

        Uses self._text_proc_cursor to track how far into self.accumulated_text
        we have already scanned, so each <tool_call>...</tool_call> block is
        emitted exactly once even as new tokens keep arriving after it.
        """
        if not self.tools:
            clean_delta = self._clean_text_delta(text_delta)
            if clean_delta:
                await self.send_json({"type": "response.text.delta", "delta": clean_delta})
            return

        self.accumulated_text += text_delta
        unprocessed = self.accumulated_text[self._text_proc_cursor:]

        if not self.in_tool_call:
            tc_pos = unprocessed.find("<tool_call>")
            if tc_pos != -1:
                content_before = unprocessed[:tc_pos]
                clean_content = self._clean_text_delta(content_before)
                if clean_content:
                    await self.send_json({"type": "response.text.delta", "delta": clean_content})
                self.in_tool_call = True
                self.current_tool_call_id = f"call_{uuid4().hex[:24]}"
                logger.debug("Tool call started")
            else:
                clean_delta = self._clean_text_delta(text_delta)
                if clean_delta:
                    await self.send_json({"type": "response.text.delta", "delta": clean_delta})

        if self.in_tool_call:
            unprocessed = self.accumulated_text[self._text_proc_cursor:]
            if "</tool_call>" in unprocessed:
                tc_abs_start = self._text_proc_cursor + unprocessed.find("<tool_call>")
                tc_abs_end = (
                    self._text_proc_cursor
                    + unprocessed.find("</tool_call>")
                    + len("</tool_call>")
                )
                tool_call_block = self.accumulated_text[tc_abs_start:tc_abs_end]
                parsed_call = self._parse_tool_call(tool_call_block)
                if parsed_call:
                    args_json = json.dumps(parsed_call["arguments"])
                    await self.send_json(
                        {
                            "type": "response.function_call_arguments.delta",
                            "call_id": self.current_tool_call_id,
                            "name": parsed_call["name"],
                            "delta": args_json,
                        }
                    )
                    await self.send_json(
                        {
                            "type": "response.function_call_arguments.done",
                            "call_id": self.current_tool_call_id,
                            "name": parsed_call["name"],
                            "arguments": args_json,
                        }
                    )
                    self.current_tool_calls.append(
                        {
                            "id": self.current_tool_call_id,
                            "name": parsed_call["name"],
                            "arguments": parsed_call["arguments"],
                        }
                    )
                    logger.info("Tool call completed: %s", parsed_call["name"])

                self._text_proc_cursor = tc_abs_end
                self.in_tool_call = False
                self.current_tool_call_id = None
                self.current_function_name = None
                self.tool_call_buffer = ""

                remaining = self.accumulated_text[self._text_proc_cursor:]
                clean_remaining = self._clean_text_delta(remaining)
                if clean_remaining:
                    await self.send_json({"type": "response.text.delta", "delta": clean_remaining})

    def _clean_text_delta(self, text: str) -> str:
        clean = text
        for special_token in [
            "<|im_start|>", "<|im_end|>", "<|audio_start|>",
            "<|audio_end|>", "<|audio_pad|>", "<|endoftext|>",
        ]:
            clean = clean.replace(special_token, "")
        return clean.strip()

    # -------------------------------------------------------------------------
    # WebSocket send
    # -------------------------------------------------------------------------

    async def send_json(self, payload: dict):
        await self.websocket.send_text(json.dumps(payload))
