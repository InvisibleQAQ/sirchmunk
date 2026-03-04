# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import openai
from openai import AsyncOpenAI, OpenAI

from sirchmunk.utils import LogCallback, create_logger

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Transient error types that warrant automatic retry.
# - APIConnectionError / APITimeoutError: network-level failures.
# - InternalServerError: covers all 5xx (500, 502, 503, 504).
# - RateLimitError (429): provider throttling.
# - NotFoundError (404): many third-party OpenAI-compatible providers
#   return 404 during transient routing / load-balancer hiccups.
_RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,   # all 5xx
    openai.RateLimitError,        # 429
    openai.NotFoundError,         # 404 — transient on some providers
)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0   # seconds
_DEFAULT_MAX_DELAY = 10.0   # seconds


@dataclass
class OpenAIChatResponse:
    """
    Data class representing the response from the OpenAI Chat API.
    """
    content: str
    role: str = "assistant"
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = None
    finish_reason: str = None
    logprobs: Any = None

    def __str__(self):
        return self.content

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response to a dictionary.

        Returns:
            Dict[str, Any]: The response as a dictionary.
        """
        return {
            "content": self.content,
            "role": self.role,
            "usage": self.usage,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "logprobs": self.logprobs,
        }


class OpenAIChat:
    """
    A client for interacting with OpenAI's chat completion API.
    """

    def __init__(
            self,
            api_key: str = None,
            base_url: str = None,
            model: str = None,
            log_callback: LogCallback = None,
            max_retries: int = _DEFAULT_MAX_RETRIES,
            retry_base_delay: float = _DEFAULT_BASE_DELAY,
            retry_max_delay: float = _DEFAULT_MAX_DELAY,
            **kwargs,
    ):
        """
        Initialize the OpenAIChat client.

        Args:
            api_key (str): The API key for OpenAI.
            base_url (str): The base URL for the OpenAI API.
            model (str): The model to use for chat completions.
            log_callback (LogCallback): Optional callback for logging.
            max_retries: Maximum number of retries for transient API errors.
            retry_base_delay: Initial backoff delay in seconds (doubled each retry).
            retry_max_delay: Upper bound on backoff delay in seconds.
            **kwargs: Additional keyword arguments passed to the OpenAI client create method.
        """
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self._model = model
        self._kwargs = kwargs
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay

        self._logger = create_logger(log_callback=log_callback, enable_async=False)
        self._logger_async = create_logger(log_callback=log_callback, enable_async=True)

    def _build_request_kwargs(
            self,
            stream: bool,
            enable_thinking: Optional[bool],
            **kwargs,
    ) -> Dict[str, Any]:
        """Merge instance-level and call-level kwargs for the API request.

        Precedence (highest wins): call-level kwargs > instance-level self._kwargs.
        ``extra_body`` dicts are deep-merged so that fields from both levels coexist.
        """
        request_kwargs = {**self._kwargs, **kwargs}

        extra_body = {
            **(self._kwargs.get("extra_body") or {}),
            **(kwargs.get("extra_body") or {}),
        }
        if enable_thinking is not None:
            extra_body["enable_thinking"] = enable_thinking
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        if stream and "stream_options" not in request_kwargs:
            request_kwargs["stream_options"] = {"include_usage": True}

        return request_kwargs

    def _backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff with jitter for retry *attempt* (0-based)."""
        import random
        delay = min(self._retry_base_delay * (2 ** attempt), self._retry_max_delay)
        return delay * (0.5 + random.random() * 0.5)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return True if *exc* is a transient error worth retrying."""
        return isinstance(exc, _RETRYABLE_ERRORS)

    def chat(
            self,
            messages: List[Dict[str, Any]],
            stream: bool = True,
            enable_thinking: Optional[bool] = False,
            **kwargs,
    ) -> OpenAIChatResponse:
        """
        Generate a chat completion synchronously.

        Automatically retries on transient API errors (404 from provider
        routing hiccups, 429 rate-limits, 5xx server errors, and connection
        failures) with exponential backoff.

        Args:
            messages (List[Dict[str, Any]]): A list of messages for the chat.
            stream (bool): Whether to stream the response.
            enable_thinking (Optional[bool]): Whether to enable model thinking/reasoning.
                Sent via ``extra_body``. Defaults to False. Pass None to omit.
            **kwargs: Additional keyword arguments merged with instance-level kwargs
                and forwarded to the OpenAI API. Call-level kwargs take precedence.

        Returns:
            OpenAIChatResponse: The structured response containing content, usage, etc.
        """
        request_kwargs = self._build_request_kwargs(stream, enable_thinking, **kwargs)
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                return self._do_chat(messages, stream, request_kwargs)
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc) or attempt >= self._max_retries:
                    raise
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "[LLM] chat() attempt %d/%d failed (%s: %s), retrying in %.1fs",
                    attempt + 1, self._max_retries + 1,
                    type(exc).__name__, exc, delay,
                )
                time.sleep(delay)

        raise last_exc  # unreachable, but keeps type checkers happy

    def _do_chat(
            self,
            messages: List[Dict[str, Any]],
            stream: bool,
            request_kwargs: Dict[str, Any],
    ) -> OpenAIChatResponse:
        """Single-attempt synchronous chat completion (no retry)."""
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, stream=stream, **request_kwargs
        )

        res_content: str = ""
        role: str = "assistant"
        usage: Dict[str, int] = {}
        finish_reason: str = None
        response_model: str = self._model

        if stream:
            for chunk in resp:
                if chunk.usage:
                    usage = chunk.usage.model_dump()

                if chunk.model:
                    response_model = chunk.model

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if delta.role:
                    role = delta.role
                    self._logger.info(f"[role={delta.role}] ", end="", flush=True)

                if delta.content:
                    self._logger.info(delta.content, end="", flush=True)
                    res_content += delta.content

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            self._logger.info("", end="\n", flush=True)

        else:
            message = resp.choices[0].message
            res_content = message.content or ""
            role = message.role
            finish_reason = resp.choices[0].finish_reason
            response_model = resp.model
            if resp.usage:
                usage = resp.usage.model_dump()

            self._logger.info(f"[role={role}] {res_content}")

        return OpenAIChatResponse(
            content=res_content,
            role=role,
            usage=usage,
            model=response_model,
            finish_reason=finish_reason
        )

    async def achat(
            self,
            messages: List[Dict[str, Any]],
            stream: bool = True,
            enable_thinking: Optional[bool] = False,
            **kwargs,
    ) -> OpenAIChatResponse:
        """
        Generate a chat completion asynchronously.

        Automatically retries on transient API errors with exponential
        backoff (same policy as :meth:`chat`).

        Args:
            messages (List[Dict[str, Any]]): A list of messages for the chat.
            stream (bool): Whether to stream the response.
            enable_thinking (Optional[bool]): Whether to enable model thinking/reasoning.
                Sent via ``extra_body``. Defaults to False. Pass None to omit.
            **kwargs: Additional keyword arguments merged with instance-level kwargs
                and forwarded to the OpenAI API. Call-level kwargs take precedence.

        Returns:
            OpenAIChatResponse: The structured response containing content, usage, etc.
        """
        request_kwargs = self._build_request_kwargs(stream, enable_thinking, **kwargs)
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                return await self._do_achat(messages, stream, request_kwargs)
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable(exc) or attempt >= self._max_retries:
                    raise
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "[LLM] achat() attempt %d/%d failed (%s: %s), retrying in %.1fs",
                    attempt + 1, self._max_retries + 1,
                    type(exc).__name__, exc, delay,
                )
                await asyncio.sleep(delay)

        raise last_exc  # unreachable, but keeps type checkers happy

    async def _do_achat(
            self,
            messages: List[Dict[str, Any]],
            stream: bool,
            request_kwargs: Dict[str, Any],
    ) -> OpenAIChatResponse:
        """Single-attempt asynchronous chat completion (no retry)."""
        resp = await self._async_client.chat.completions.create(
            model=self._model, messages=messages, stream=stream, **request_kwargs
        )

        res_content: str = ""
        role: str = "assistant"
        usage: Dict[str, int] = {}
        finish_reason: str = None
        response_model: str = self._model

        if stream:
            async for chunk in resp:
                if chunk.usage:
                    usage = chunk.usage.model_dump()

                if chunk.model:
                    response_model = chunk.model

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if delta.role:
                    role = delta.role
                    await self._logger_async.info(f"[role={delta.role}] ", end="", flush=True)

                if delta.content:
                    await self._logger_async.info(delta.content, end="", flush=True)
                    res_content += delta.content

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            await self._logger_async.info("", end="\n", flush=True)

        else:
            message = resp.choices[0].message
            res_content = message.content or ""
            role = message.role
            finish_reason = resp.choices[0].finish_reason
            response_model = resp.model
            if resp.usage:
                usage = resp.usage.model_dump()

            await self._logger_async.info(f"[role={role}] {res_content}")

        return OpenAIChatResponse(
            content=res_content,
            role=role,
            usage=usage,
            model=response_model,
            finish_reason=finish_reason
        )
