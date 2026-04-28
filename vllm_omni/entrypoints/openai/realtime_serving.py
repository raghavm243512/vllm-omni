# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extended OpenAIServingRealtime with conversation context support."""

import asyncio
from collections.abc import AsyncGenerator
from typing import cast

import numpy as np
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.openai.realtime.serving import (
    OpenAIServingRealtime as VllmOpenAIServingRealtime,
)
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.renderers.inputs.preprocess import parse_model_prompt

logger = init_logger(__name__)


class OpenAIServingRealtime(VllmOpenAIServingRealtime):
    """Extended realtime serving with conversation context support."""

    def __init__(self, *args, speech_service=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Optional TTS service for generating audio responses
        self.speech_service = speech_service

    async def transcribe_realtime(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        conversation_context: str | None = None,
    ) -> AsyncGenerator[StreamingInput, None]:
        """Transform audio stream into StreamingInput for engine.generate().

        Args:
            audio_stream: Async generator yielding float32 numpy audio arrays
            input_stream: Queue containing context token IDs from previous
                generation outputs. Used for autoregressive multi-turn
                processing where each generation's output becomes the context
                for the next iteration.
            conversation_context: Optional conversation history and tool definitions
                to inject into the prompt.

        Yields:
            StreamingInput objects containing audio prompts for the engine
        """
        model_config = self.model_config
        renderer = self.renderer

        # Call buffer_realtime_audio with conversation_context
        stream_input_iter = cast(
            AsyncGenerator[PromptType, None],
            self.model_cls.buffer_realtime_audio(
                audio_stream, input_stream, model_config, conversation_context
            ),
        )

        async for prompt in stream_input_iter:
            parsed_prompt = parse_model_prompt(model_config, prompt)
            (engine_prompt,) = await renderer.render_cmpl_async([parsed_prompt])

            yield StreamingInput(prompt=engine_prompt)
