# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extended protocol events for OpenAI-compatible realtime API with tool calling."""

from typing import Literal

from pydantic import Field
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel

# Re-export base events from vLLM
from vllm.entrypoints.openai.realtime.protocol import (  # noqa: F401
    ErrorEvent,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    SessionCreated,
    SessionUpdate,
)

# Additional Server -> Client Events for Tool Calling


class ResponseTextDelta(OpenAIBaseModel):
    """Incremental text response (for debugging and tool calls)"""

    type: Literal["response.text.delta"] = "response.text.delta"
    delta: str  # Incremental text


class ResponseTextDone(OpenAIBaseModel):
    """Final text response"""

    type: Literal["response.text.done"] = "response.text.done"
    text: str  # Complete text


class ResponseFunctionCallArgumentsDelta(OpenAIBaseModel):
    """Incremental function call arguments"""

    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"
    call_id: str = Field(description="Unique ID for this function call")
    name: str = Field(description="Function name being called")
    delta: str  # Incremental JSON arguments


class ResponseFunctionCallArgumentsDone(OpenAIBaseModel):
    """Complete function call arguments"""

    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    call_id: str = Field(description="Unique ID for this function call")
    name: str = Field(description="Function name being called")
    arguments: str  # Complete JSON arguments


class ResponseAudioDelta(OpenAIBaseModel):
    """Incremental audio response"""

    type: Literal["response.audio.delta"] = "response.audio.delta"
    audio: str = Field(description="Base64-encoded audio chunk")
    sample_rate_hz: int = Field(default=24000, description="Audio sample rate")


class ResponseAudioDone(OpenAIBaseModel):
    """Audio response complete"""

    type: Literal["response.audio.done"] = "response.audio.done"
    has_audio: bool = True


# Client -> Server Events for Tool Results


class ConversationItemMessage(OpenAIBaseModel):
    """Conversation item for regular messages"""

    type: Literal["message"] = "message"
    role: str
    content: str


class ConversationItemFunctionCallOutput(OpenAIBaseModel):
    """Conversation item for function call results"""

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str = Field(description="ID of the function call this is a result for")
    output: str = Field(description="JSON-encoded function result")


class ConversationItemCreate(OpenAIBaseModel):
    """Create a new conversation item (e.g., tool result)"""

    type: Literal["conversation.item.create"] = "conversation.item.create"
    item: ConversationItemMessage | ConversationItemFunctionCallOutput
