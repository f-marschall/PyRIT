# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import TYPE_CHECKING, Any, Optional

import httpx

from pyrit.exceptions import EmptyResponseException, pyrit_target_retry
from pyrit.identifiers import TargetIdentifier
from pyrit.models import (
    Message,
    construct_response_from_request,
)
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.prompt_target.common.utils import limit_requests_per_minute

if TYPE_CHECKING:
    from a2a.client import ClientConfig
    from a2a.types import (
        Message as A2AMessage,
    )
    from a2a.types import (
        MessageSendConfiguration,
        Task,
    )

logger = logging.getLogger(__name__)


class A2AChatTarget(PromptChatTarget):
    """
    A prompt chat target for interacting with remote agents via the A2A (Agent-to-Agent) protocol.

    The A2A protocol defines a standard way for agents to communicate.
    This target wraps the official ``a2a-sdk`` Python package, using ``ClientFactory``
    to resolve the agent card and select the appropriate transport (JSON-RPC, HTTP+JSON, or gRPC).

    For the full specification see https://a2a-protocol.org/latest/specification/

    This target supports:
    - Sending text prompts via the SDK ``Client.send_message`` method
    - Automatic transport negotiation based on the agent card
    - Blocking mode where the server waits for task completion before responding
    - Polling mode via the SDK for asynchronous task completion
    - Parsing text from both direct Message responses and Task artifacts
    - Multi-turn conversations via A2A contextId and taskId
    """

    DEFAULT_ACCEPTED_OUTPUT_MODES: list[str] = ["text/plain", "application/json"]

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: Optional[str] = None,
        auth_header_name: str = "Authorization",
        auth_header_value_prefix: str = "Bearer ",
        preferred_transport: str = "JSONRPC",
        use_blocking: bool = True,
        streaming: bool = False,
        accepted_output_modes: Optional[list[str]] = None,
        max_requests_per_minute: Optional[int] = None,
        model_name: str = "",
        httpx_client_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the A2A Chat Target.

        Args:
            endpoint (str): The base URL of the A2A-compliant agent
                (e.g., ``https://agent.example.com``). The SDK resolves the agent card
                from the ``/.well-known/agent.json`` path automatically.
            api_key (str, Optional): API key for authentication. Sent using the configured
                auth header. Defaults to None.
            auth_header_name (str): HTTP header name for authentication.
                Defaults to ``Authorization``.
            auth_header_value_prefix (str): Prefix prepended to the API key in the auth
                header value (e.g., ``Bearer ``). Defaults to ``Bearer ``.
            preferred_transport (str): The preferred A2A transport protocol. One of
                ``JSONRPC``, ``HTTP+JSON``, or ``GRPC``. The SDK will negotiate the actual
                transport based on the agent card capabilities. Defaults to ``JSONRPC``.
            use_blocking (bool): Whether to set ``blocking: true`` in the message
                configuration so the server waits for completion. When False, the SDK
                uses polling. Defaults to True.
            streaming (bool): Whether to use SSE streaming for responses.
                Defaults to False.
            accepted_output_modes (list[str], Optional): MIME types the client accepts.
                Defaults to ``["text/plain", "application/json"]``.
            max_requests_per_minute (int, Optional): Rate limit. Defaults to None.
            model_name (str): Model name for identification. Defaults to empty string.
            httpx_client_kwargs (dict, Optional): Extra keyword arguments passed to
                ``httpx.AsyncClient``. Defaults to None.

        Raises:
            RuntimeError: If the a2a-sdk package is not installed.
        """
        super().__init__(
            max_requests_per_minute=max_requests_per_minute,
            endpoint=endpoint,
            model_name=model_name,
        )

        try:
            import a2a.client  # noqa: F401
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Could not import a2a-sdk. Install it via 'pip install pyrit[a2a]'"
            ) from e

        self._api_key = api_key
        self._auth_header_name = auth_header_name
        self._auth_header_value_prefix = auth_header_value_prefix
        self._preferred_transport = preferred_transport
        self._use_blocking = use_blocking
        self._streaming = streaming
        self._accepted_output_modes = accepted_output_modes or self.DEFAULT_ACCEPTED_OUTPUT_MODES
        self._httpx_client_kwargs = httpx_client_kwargs or {}

        # Maps PyRIT conversation_id -> A2A contextId for multi-turn support
        self._conversation_context_map: dict[str, str] = {}
        # Maps PyRIT conversation_id -> last A2A taskId
        self._conversation_task_map: dict[str, str] = {}

    def _build_identifier(self) -> TargetIdentifier:
        """
        Build the identifier with A2A-specific parameters.

        Returns:
            TargetIdentifier: The identifier for this target instance.
        """
        return self._create_identifier(
            target_specific_params={
                "preferred_transport": self._preferred_transport,
                "use_blocking": self._use_blocking,
                "streaming": self._streaming,
            },
        )

    def is_json_response_supported(self) -> bool:
        """
        A2A does not natively support forcing JSON response format.

        Returns:
            bool: Always False.
        """
        return False

    def _build_httpx_client(self) -> httpx.AsyncClient:
        """
        Build an httpx.AsyncClient with optional auth headers.

        Returns:
            httpx.AsyncClient: The configured HTTP client.
        """
        headers = dict(self._httpx_client_kwargs.pop("headers", {}))
        if self._api_key:
            headers[self._auth_header_name] = f"{self._auth_header_value_prefix}{self._api_key}"
        return httpx.AsyncClient(headers=headers, **self._httpx_client_kwargs)

    def _build_client_config(self, *, httpx_client: httpx.AsyncClient) -> "ClientConfig":
        """
        Build the a2a-sdk ClientConfig.

        Args:
            httpx_client (httpx.AsyncClient): The HTTP client to use.

        Returns:
            ClientConfig: The SDK client configuration.
        """
        from a2a.client import ClientConfig

        return ClientConfig(
            httpx_client=httpx_client,
            streaming=self._streaming,
            polling=not self._use_blocking,
            supported_transports=[self._preferred_transport],
            accepted_output_modes=self._accepted_output_modes,
        )

    @staticmethod
    def _extract_text_from_task(task: "Task") -> str:
        """
        Extract text content from an A2A Task object.

        Looks for text in artifacts first, then falls back to status message,
        and finally to history messages.

        Args:
            task (Task): An A2A Task object from the SDK.

        Returns:
            str: The extracted text content.

        Raises:
            EmptyResponseException: If no text content is found.
        """
        from a2a.types import Role, TextPart

        text_parts: list[str] = []

        # Extract from artifacts (primary output)
        for artifact in task.artifacts or []:
            for part in artifact.parts:
                if isinstance(part.root, TextPart):
                    text_parts.append(part.root.text)

        # Extract from status message (agent feedback)
        if task.status and task.status.message and task.status.message.role == Role.agent:
            for part in task.status.message.parts:
                if isinstance(part.root, TextPart):
                    text_parts.append(part.root.text)

        # Extract from history messages
        for msg in task.history or []:
            if msg.role == Role.agent:
                for part in msg.parts:
                    if isinstance(part.root, TextPart):
                        text_parts.append(part.root.text)

        if not text_parts:
            raise EmptyResponseException(message="A2A Task response contained no text content.")

        return "\n".join(text_parts)

    @staticmethod
    def _extract_text_from_a2a_message(a2a_message: "A2AMessage") -> str:
        """
        Extract text content from a direct A2A Message response.

        Args:
            a2a_message (A2AMessage): An A2A Message object from the SDK.

        Returns:
            str: The extracted text content.

        Raises:
            EmptyResponseException: If no text content is found.
        """
        from a2a.types import TextPart

        text_parts: list[str] = []

        for part in a2a_message.parts:
            if isinstance(part.root, TextPart):
                text_parts.append(part.root.text)

        if not text_parts:
            raise EmptyResponseException(message="A2A Message response contained no text content.")

        return "\n".join(text_parts)

    def _update_conversation_maps(
        self,
        *,
        conversation_id: str,
        task: Optional["Task"] = None,
        a2a_message: Optional["A2AMessage"] = None,
    ) -> None:
        """
        Update conversation-to-A2A-ID mappings for multi-turn support.

        Args:
            conversation_id (str): The PyRIT conversation ID.
            task (Task, Optional): An A2A Task if the response was a task. Defaults to None.
            a2a_message (A2AMessage, Optional): An A2A Message if it was a direct reply.
                Defaults to None.
        """
        if task:
            self._conversation_task_map[conversation_id] = task.id
            if task.context_id:
                self._conversation_context_map[conversation_id] = task.context_id
        elif a2a_message and a2a_message.context_id:
            self._conversation_context_map[conversation_id] = a2a_message.context_id

    # ------------------------------------------------------------------
    # Core send implementation
    # ------------------------------------------------------------------

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, message: Message) -> list[Message]:
        """
        Send a prompt to an A2A-compliant agent using the official a2a-sdk.

        Resolves the agent card from the endpoint, creates a transport-aware client,
        sends the message, and parses the response. The SDK handles transport negotiation,
        blocking/polling, and streaming automatically.

        Args:
            message (Message): The message to send, containing one text message piece.

        Returns:
            list[Message]: A list containing the response message from the A2A agent.

        Raises:
            RuntimeError: If the A2A agent returns an error or the task fails/is canceled/rejected.
        """
        self._validate_request(message=message)

        request_piece = message.message_pieces[0]
        prompt_text = request_piece.converted_value
        conversation_id = request_piece.conversation_id

        response_text = await self._send_via_sdk_async(
            prompt_text=prompt_text,
            conversation_id=conversation_id,
        )

        response_message = construct_response_from_request(
            request=request_piece,
            response_text_pieces=[response_text],
        )
        return [response_message]

    async def _send_via_sdk_async(
        self,
        *,
        prompt_text: str,
        conversation_id: str,
    ) -> str:
        """
        Send a message using the a2a-sdk Client and return the extracted response text.

        Args:
            prompt_text (str): The text content to send.
            conversation_id (str): The PyRIT conversation ID for multi-turn tracking.

        Returns:
            str: The extracted response text from the agent.

        Raises:
            RuntimeError: If the task fails, is canceled, or is rejected.
        """
        # Build multi-turn A2A message
        from a2a.client import ClientFactory, create_text_message_object
        from a2a.types import MessageSendConfiguration, Role

        a2a_msg = create_text_message_object(role=Role.user, content=prompt_text)

        existing_task_id = self._conversation_task_map.get(conversation_id)
        existing_context_id = self._conversation_context_map.get(conversation_id)
        if existing_task_id:
            a2a_msg.task_id = existing_task_id
        if existing_context_id:
            a2a_msg.context_id = existing_context_id

        logger.info("Sending A2A message to %s", self._endpoint)

        httpx_client = self._build_httpx_client()
        config = self._build_client_config(httpx_client=httpx_client)

        client = await ClientFactory.connect(
            agent=self._endpoint,
            client_config=config,
        )

        configuration = MessageSendConfiguration(
            accepted_output_modes=self._accepted_output_modes,
            blocking=self._use_blocking,
        )

        return await self._collect_response_async(
            client=client,
            a2a_msg=a2a_msg,
            configuration=configuration,
            conversation_id=conversation_id,
        )

    async def _collect_response_async(
        self,
        *,
        client: Any,
        a2a_msg: "A2AMessage",
        configuration: "MessageSendConfiguration",
        conversation_id: str,
    ) -> str:
        """
        Iterate over the SDK client's response stream and extract the final text.

        Args:
            client: The a2a-sdk Client instance.
            a2a_msg (A2AMessage): The message being sent.
            configuration (MessageSendConfiguration): Per-request config overrides.
            conversation_id (str): The PyRIT conversation ID.

        Returns:
            str: The extracted response text.

        Raises:
            RuntimeError: If the task reaches a terminal failure state.
            EmptyResponseException: If the response contains no text.
        """
        from a2a.types import Message as A2AMessage
        from a2a.types import Task

        final_task: Optional[Task] = None
        final_message: Optional[A2AMessage] = None

        async for event in client.send_message(a2a_msg, configuration=configuration):
            if isinstance(event, A2AMessage):
                final_message = event
            elif isinstance(event, tuple):
                task, _update_event = event
                if isinstance(task, Task):
                    final_task = task

        if final_message:
            self._update_conversation_maps(conversation_id=conversation_id, a2a_message=final_message)
            return self._extract_text_from_a2a_message(final_message)

        if final_task:
            self._check_task_terminal_state(task=final_task)
            self._update_conversation_maps(conversation_id=conversation_id, task=final_task)
            return self._extract_text_from_task(final_task)

        raise EmptyResponseException(message="A2A agent returned no response events.")

    @staticmethod
    def _check_task_terminal_state(*, task: "Task") -> None:
        """
        Raise an error if the task reached a failure terminal state.

        Args:
            task (Task): The A2A Task to check.

        Raises:
            RuntimeError: If the task failed, was canceled, rejected, or requires auth.
        """
        from a2a.types import TaskState

        state = task.status.state
        if state == TaskState.failed:
            raise RuntimeError(f"A2A task failed: {task.status.message}")
        if state == TaskState.canceled:
            raise RuntimeError("A2A task was canceled by the remote agent.")
        if state == TaskState.rejected:
            raise RuntimeError("A2A task was rejected by the remote agent.")
        if state == TaskState.auth_required:
            raise RuntimeError("A2A task requires authentication to proceed.")

    def _validate_request(self, *, message: Message) -> None:
        """
        Validate the message before sending.

        The A2A target currently supports a single text message piece per request.

        Args:
            message (Message): The message to validate.

        Raises:
            ValueError: If the message has no pieces, more than one piece,
                or contains non-text content.
        """
        pieces = message.message_pieces
        if not pieces:
            raise ValueError("Message must contain at least one message piece.")

        if len(pieces) != 1:
            raise ValueError(
                f"A2AChatTarget currently supports a single message piece per request. "
                f"Received: {len(pieces)} pieces."
            )

        piece = pieces[0]
        if piece.converted_value_data_type != "text":
            raise ValueError(
                f"A2AChatTarget only supports text prompts. "
                f"Received data type: {piece.converted_value_data_type}"
            )

    async def cleanup_conversation_async(self, *, conversation_id: str) -> None:
        """
        Remove stored A2A mappings for a conversation.

        Resets multi-turn state so subsequent messages start a fresh A2A context.

        Args:
            conversation_id (str): The conversation ID to clean up.
        """
        self._conversation_context_map.pop(conversation_id, None)
        self._conversation_task_map.pop(conversation_id, None)
