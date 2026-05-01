# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extended RealtimeConnection with tool calling support for vLLM Omni."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncGenerator
from typing import cast
from uuid import uuid4

import numpy as np
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel, UsageInfo
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
from vllm_omni.entrypoints.openai.realtime_protocol import (
    RealtimeEventType,
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseFunctionCallArgumentsDelta,
    ResponseFunctionCallArgumentsDone,
    ResponseTextDelta,
    ResponseTextDone,
)
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

        # Tool call parsing state
        self.in_tool_call = False
        self.current_tool_calls: list[dict] = []
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

        # Index into conversation_items marking the boundary between prior turns
        # and the current turn. Captured at the start of every turn so the
        # tool-context audio pass can frame prior history and current-turn tool
        # interaction separately, keeping the temporal layout clear to the model.
        self._turn_start_idx: int = 0

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

        if event_type == RealtimeEventType.SESSION_UPDATE:
            session = event.get("session", {})
            self.tools = session.get("tools")
            self.instructions = session.get("instructions")
            logger.info(f"Session updated with {len(self.tools) if self.tools else 0} tools")
            await super().handle_event(event)

        elif event_type == RealtimeEventType.INPUT_AUDIO_BUFFER_COMMIT:
            # Override commit handling: start generation AND close the audio stream
            # so buffer_realtime_audio() can flush and finish, which in turn lets
            # _add_streaming_input_request send resumable=False to the engine.
            # Without the None sentinel, the thinker stage never gets finished=True
            # and never forwards to talker→code2wav, so audio never arrives.
            commit_event = InputAudioBufferCommit(**event)
            if not commit_event.final:
                await self.start_generation()
            self.audio_queue.put_nowait(None)

        elif event_type == RealtimeEventType.CONVERSATION_ITEM_CREATE:
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
            self._run_audio_from_tool_context(append_response=True)
        )

    # -------------------------------------------------------------------------
    # Generation entry point
    # -------------------------------------------------------------------------

    async def start_generation(self):
        """Start the transcription generation task with conversation context support."""
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        # Snapshot boundary first, then append the user placeholder so it lands
        # at index _turn_start_idx: inside current_items for this turn (filtered
        # out by current_tool_items) and inside prior_items for future turns.
        self._turn_start_idx = len(self.conversation_items)
        self.conversation_items.append({"role": "user", "content": None})

        if self.instructions and not self.tools:
            # Instructions only — no tool calls possible, skip the text-only thinker
            # pass entirely. Drain audio into _cached_user_audio then run the audio
            # pass directly with the system instruction injected by
            # _run_audio_from_tool_context.
            self.generation_task = asyncio.create_task(self._drain_and_run_audio())
            return

        audio_stream = self.audio_stream_generator()
        input_stream = asyncio.Queue[list[int]]()

        # IMPORTANT: Only inject conversation_context / prior_blocks when tools are
        # configured. Audio-only passes MUST receive context=None because
        # buffer_realtime_audio splits a non-None context into a separate initial
        # add_request, leaving audio tokens as streaming updates. That makes
        # thinker_output.prompt_token_ids contain only system tokens —
        # _compute_talker_prompt_ids_length finds no <|im_start|>user marker,
        # returns 0, and the talker hits the decode path instead of prefill,
        # crashing with "Missing prefill_consumed_text_tokens".
        if self.tools:
            conversation_context = self._build_system_context()
            prior_blocks = self._render_prior_blocks()
        else:
            conversation_context = None
            prior_blocks = None

        streaming_input_gen = self.serving.transcribe_realtime(
            audio_stream, input_stream, conversation_context, prior_blocks
        )

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
            await self.send(ResponseAudioDelta(
                audio=base64.b64encode(piece).decode("utf-8"),
                sample_rate_hz=sample_rate,
            ))

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

        t_start = time.monotonic()
        try:
            if self.tools:
                # --- Single combined text+audio pass with tool detection ---
                # output_modalities=["text", "audio"]: in the dual-stage architecture,
                # text tokens (stage 0 / thinker) arrive before audio (stage 2 / code2wav).
                # We watch text for <tool_call> and abort cleanly before any audio is sent.
                # For no-tool turns this avoids a sequential thinker pass followed by a
                # full re-run audio pass — audio just flows through in one shot.
                result_gen = self.engine.generate(
                    prompt=streaming_input_gen,
                    sampling_params_list=coerce_param_message_types(
                        list(self.engine.default_sampling_params_list), is_streaming=True
                    ),
                    request_id=request_id,
                    output_modalities=["text", "audio"],
                )
                async for output in result_gen:
                    if output.outputs:
                        token_ids = list(output.outputs[0].token_ids)
                        if token_ids:
                            text_delta = self._decode_tokens(token_ids)
                            if text_delta:
                                await self._process_text_delta(text_delta)
                    # Abort as soon as a complete tool call is detected. Text arrives
                    # before audio in the pipeline, so this is always a clean abort
                    # with no audio sent.
                    if self.current_tool_calls:
                        logger.info(
                            "[TIMING] Tool call detected at %.2fs — aborting before audio",
                            time.monotonic() - t_start,
                        )
                        await self.engine.abort(request_id)
                        break
                    audio_chunks, sample_rate = self._extract_audio_chunks(output)
                    for chunk in audio_chunks:
                        if not sent_audio:
                            logger.info(
                                "[TIMING] First audio chunk at %.2fs (no tool call)",
                                time.monotonic() - t_start,
                            )
                        sent_audio = True
                        await self._send_audio_delta(chunk, sample_rate)
                    if not self._is_connected:
                        break

                logger.info(
                    "[TIMING] Generation loop finished: %.2fs | tool_calls=%d | sent_audio=%s",
                    time.monotonic() - t_start,
                    len(self.current_tool_calls),
                    sent_audio,
                )
                logger.info("[TEXT] Raw thinker output: %r", self.accumulated_text)

                if self.current_tool_calls:
                    # Thinking text and raw <tool_call> XML are noise in history.
                    assistant_content = None
                else:
                    raw = self._strip_thinking(self.accumulated_text)
                    assistant_content = self._clean_text_delta(raw) or None
                    logger.info("[TEXT] Visible assistant text: %r", assistant_content)

                self.conversation_items.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": self.current_tool_calls or None,
                    }
                )

                if self.current_tool_calls:
                    logger.info(
                        "Emitted %d tool call(s); waiting for client tool responses",
                        len(self.current_tool_calls),
                    )
                    visible_text = self._clean_text_delta(self._strip_thinking(self.accumulated_text))
                    if visible_text:
                        await self.send(ResponseTextDone(text=visible_text))
                    if self._pending_tool_context:
                        self._pending_tool_context = False
                        logger.info("Tool result was already received — starting audio pass now")
                        self.generation_task = asyncio.create_task(
                            self._run_audio_from_tool_context(append_response=True)
                        )
                    return

                # No tool calls: audio was already sent inline above. Fall through
                # to the sent_audio / ResponseAudioDone block at the end of the try.

            else:
                # No tools, no instructions — single audio pass (fast path)
                result_gen = self.engine.generate(
                    prompt=streaming_input_gen,
                    request_id=request_id,
                    output_modalities=["audio"],
                    sampling_params_list=coerce_param_message_types(list(self.engine.default_sampling_params_list), is_streaming=True),
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
                await self.send(ResponseAudioDone())
                done_sent = True

        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            if self._is_connected and not done_sent and sent_audio:
                try:
                    await self.send(ResponseAudioDone())
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            # Drain any unconsumed audio into the cache. If the engine was aborted
            # before consuming all audio (e.g. tool call detected mid-stream),
            # audio_stream_generator only populated _cached_user_audio up to the
            # abort point. Drain the rest here so the audio pass has the full clip.
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get_nowait()
                if chunk is not None:
                    self._cached_user_audio.append(chunk)

    async def _drain_and_run_audio(self) -> None:
        """Consume audio queue into cache then run audio pass (instructions-only path).

        Bypasses the text-only thinker pass when there are no tools — no tool
        calls are possible so the extra pass would only add latency.
        """
        self._cached_user_audio = []
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                break
            self._cached_user_audio.append(chunk)
        await self._run_audio_from_tool_context(append_response=True)

    async def _run_audio_from_tool_context(self, append_response: bool = False) -> None:
        """Generate speech after receiving tool results (or for direct responses).

        Prompt structure (proper Qwen3 chat-template format):

            <|im_start|>system
            {tools + instructions}
            <|im_end|>
            [prior turns: <assistant tool_call> + <user tool_response> blocks]
            <|im_start|>user
            <|audio_start|><|audio_pad|><|audio_end|>
            <|im_end|>
            [current turn: <assistant tool_call> + <user tool_response> blocks]
            <|im_start|>assistant

        _thinker_to_talker_prefill and _compute_talker_prompt_ids_length both skip
        system blocks, text-only user blocks, and non-last assistant blocks — only
        the audio-bearing user block and the final assistant block feed into the
        talker. The thinker still attends to all blocks for response generation.

        Args:
            append_response: When True, append an assistant conversation item with
                the spoken text after generation. Set for tool-result and
                instructions-only paths. Leave False for the no-tool-call path,
                which already has an assistant item from the thinker pass.
        """
        sent_audio = False
        done_sent = False
        spoken_text = ""
        self._realtime_audio_ref = None
        t_audio_start = time.monotonic()
        try:
            audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"

            parts: list[str] = []

            # System block: tools + instructions (skipped by talker)
            system_parts: list[str] = []
            if self.tools:
                system_parts.append(self._format_tools_for_prompt())
            if self.instructions:
                system_parts.append(self.instructions)
            if system_parts:
                system_body = "\n".join(system_parts)
                parts.append(f"<|im_start|>system\n{system_body}<|im_end|>")

            # Prior turns as proper chat-template blocks (non-last assistant
            # blocks and text-only user blocks are skipped by talker; thinker
            # attends to them for conversation context).
            prior_items = self.conversation_items[: self._turn_start_idx]
            for item in prior_items:
                block = self._render_item_as_template_block(item)
                if block:
                    parts.append(block)

            # Audio user block (included by talker for acoustic conditioning)
            parts.append(f"<|im_start|>user\n{audio_placeholder}<|im_end|>")

            # Current-turn tool interaction: assistant tool_call(s) and tool
            # result(s). These appear after the audio so the thinker has
            # temporal ordering right. Talker skips them (non-last assistant
            # block + text-only user block).
            current_items = self.conversation_items[self._turn_start_idx :]
            current_tool_items = [
                item for item in current_items
                if item.get("role") == "tool"
                or (item.get("role") == "assistant" and item.get("tool_calls"))
            ]
            for item in current_tool_items:
                block = self._render_item_as_template_block(item)
                if block:
                    parts.append(block)

            # Final assistant block (included by talker — this is where speech is generated)
            parts.append("<|im_start|>assistant\n")
            full_prompt = "\n".join(parts)

            # Use current-turn audio for acoustic conditioning so the talker's
            # hidden_projection receives full-length in-distribution features.
            ref_audio = self._cached_user_audio
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
                sampling_params_list=coerce_param_message_types(
                    list(self.engine.default_sampling_params_list), is_streaming=True
                ),
            )

            async for output in result_gen:
                if output.outputs:
                    spoken_text += output.outputs[0].text or ""
                audio_chunks, sample_rate = self._extract_audio_chunks(output)
                for chunk in audio_chunks:
                    if not sent_audio:
                        logger.info(
                            "[TIMING] tool-context audio pass first chunk at %.2fs",
                            time.monotonic() - t_audio_start,
                        )
                    sent_audio = True
                    await self._send_audio_delta(chunk, sample_rate)
                if not self._is_connected:
                    break

            logger.info(
                "[TIMING] tool-context audio pass done: %.2fs | sent_audio=%s",
                time.monotonic() - t_audio_start,
                sent_audio,
            )
            logger.info("[TEXT] Spoken text from audio pass: %r", spoken_text)
            if append_response:
                clean = self._clean_text_delta(self._strip_thinking(spoken_text))
                self.conversation_items.append({
                    "role": "assistant",
                    "content": clean or None,
                    "tool_calls": None,
                })

            if sent_audio:
                await self.send(ResponseAudioDone())
                done_sent = True

        except Exception as exc:
            logger.exception("Error in audio-from-tool-context pass: %s", exc)
            await self.send_error(str(exc), "processing_error")
        finally:
            if not done_sent and sent_audio:
                try:
                    await self.send(ResponseAudioDone())
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # Conversation helpers
    # -------------------------------------------------------------------------

    def _render_item_as_template_block(self, item: dict) -> str:
        """Render a conversation item as a proper Qwen3 chat-template block."""
        role = item.get("role", "")
        if role == "user":
            content = item.get("content") or "[User's audio]"
            return f"<|im_start|>user\n{content}<|im_end|>"
        elif role == "assistant":
            content_parts: list[str] = []
            content = item.get("content")
            if content:
                clean = self._clean_text_delta(content)
                if clean:
                    content_parts.append(clean)
            for call in (item.get("tool_calls") or []):
                args_str = json.dumps({"name": call["name"], "arguments": call.get("arguments", {})})
                content_parts.append(f"<tool_call>\n{args_str}\n</tool_call>")
            if not content_parts:
                return ""
            return f"<|im_start|>assistant\n{''.join(content_parts)}<|im_end|>"
        elif role == "tool":
            content = item.get("content", "")
            return f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>"
        return ""

    def _render_prior_blocks(self) -> str | None:
        """Render prior-turn conversation items as proper Qwen3 chat-template blocks.

        These are injected between the system block and the current audio user
        block in buffer_realtime_audio, giving the thinker full conversation
        context in the format it was trained on rather than prose in the system block.
        """
        prior_items = self.conversation_items[: self._turn_start_idx]
        blocks = [self._render_item_as_template_block(item) for item in prior_items]
        rendered = "\n".join(b for b in blocks if b)
        return rendered or None

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
        """Build the system block content: tools definition + session instructions."""
        parts: list[str] = []
        if self.tools:
            parts.append(self._format_tools_for_prompt())
        if self.instructions:
            parts.append(self.instructions)
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
            logger.warning("Failed to decode tokens: %s", e)
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
                await self.send(ResponseTextDelta(delta=clean_delta))
            return

        self.accumulated_text += text_delta
        unprocessed = self.accumulated_text[self._text_proc_cursor:]

        if not self.in_tool_call:
            tc_pos = unprocessed.find("<tool_call>")
            if tc_pos != -1:
                content_before = unprocessed[:tc_pos]
                clean_content = self._clean_text_delta(content_before)
                if clean_content:
                    await self.send(ResponseTextDelta(delta=clean_content))
                self.in_tool_call = True
                self.current_tool_call_id = f"call_{uuid4().hex[:24]}"
                logger.debug("Tool call started")
            else:
                clean_delta = self._clean_text_delta(text_delta)
                if clean_delta:
                    await self.send(ResponseTextDelta(delta=clean_delta))

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
                    await self.send(ResponseFunctionCallArgumentsDelta(
                        call_id=self.current_tool_call_id,
                        name=parsed_call["name"],
                        delta=args_json,
                    ))
                    await self.send(ResponseFunctionCallArgumentsDone(
                        call_id=self.current_tool_call_id,
                        name=parsed_call["name"],
                        arguments=args_json,
                    ))
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

                remaining = self.accumulated_text[self._text_proc_cursor:]
                clean_remaining = self._clean_text_delta(remaining)
                if clean_remaining:
                    await self.send(ResponseTextDelta(delta=clean_remaining))

    def _strip_thinking(self, text: str) -> str:
        """Strip Qwen3 <think>...</think> blocks from thinker output."""
        result = text
        while "<think>" in result and "</think>" in result:
            start = result.find("<think>")
            end = result.find("</think>") + len("</think>")
            result = result[:start] + result[end:]
        return result.strip()

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

    async def send(self, event: OpenAIBaseModel) -> None:  # type: ignore[override]
        await self.websocket.send_text(event.model_dump_json())
