from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types
from openai import AsyncOpenAI

from schemas import ChatMessage, ErrorType, LLMError, ModelResponse, Provider


def clasificar_error(e: Exception) -> ErrorType:
    """Cada SDK tiene sus propias excepciones, así que miro el status HTTP y el nombre de la clase."""
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    nombre = type(e).__name__
    texto = str(e).lower()

    if status == 429 or "RateLimit" in nombre:
        return ErrorType.RATE_LIMIT
    if status in (401, 403) or "Authentication" in nombre or "PermissionDenied" in nombre:
        return ErrorType.AUTH
    if status == 400 and "api key" in texto:  # Gemini devuelve 400 cuando la key no es válida
        return ErrorType.AUTH
    if "Timeout" in nombre:  # va antes que Connection porque APITimeoutError hereda de APIConnectionError
        return ErrorType.TIMEOUT
    if "Connect" in nombre:
        return ErrorType.CONNECTION
    if isinstance(status, int) and status >= 500:
        return ErrorType.SERVER
    return ErrorType.OTHER


class BaseLLMClient(ABC):
    """Contrato que tiene que cumplir cualquier cliente, sin importar el proveedor."""

    provider: Provider

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Respuesta completa. Si algo falla devuelve un ModelResponse con error, no tira excepción."""
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Respuesta token a token. Si falla levanta LLMError (lo atrapa el manager)."""
        raise NotImplementedError
        yield  # nunca se ejecuta, solo para que Python lo trate como generador

    def _ok(self, texto) -> ModelResponse:
        if not texto or not texto.strip():
            return self._error(LLMError("la respuesta vino vacía"))
        return ModelResponse(provider=self.provider, model=self.model, content=texto)

    def _error(self, e: Exception) -> ModelResponse:
        tipo = e.error_type if isinstance(e, LLMError) else clasificar_error(e)
        return ModelResponse(
            provider=self.provider,
            model=self.model,
            content="",
            error=f"[{tipo.value}] {e}",
            error_type=tipo,
        )


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__(api_key, model, temperature, max_tokens)
        # max_retries=0 porque los reintentos los hace el manager
        self._client = AsyncOpenAI(api_key=api_key, max_retries=0, timeout=30)

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
            return self._ok(response.choices[0].message.content)
        except Exception as e:
            return self._error(e)

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise LLMError(str(e), clasificar_error(e)) from e


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__(api_key, model, temperature, max_tokens)
        self._client = AsyncAnthropic(api_key=api_key, max_retries=0, timeout=30)

    def _preparar(self, messages: List[ChatMessage]) -> dict:
        # Anthropic no acepta role "system" dentro de messages, va como parámetro aparte.
        # La versión nueva del SDK tampoco recibe temperature, por eso no se manda.
        system = "\n".join(m.content for m in messages if m.role == "system")
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,  # acá es obligatorio
            "messages": [m.model_dump() for m in messages if m.role != "system"],
        }
        if system:
            params["system"] = system
        return params

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.messages.create(**self._preparar(messages))
            texto = "".join(b.text for b in response.content if b.type == "text")
            return self._ok(texto)
        except Exception as e:
            return self._error(e)

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            async with self._client.messages.stream(**self._preparar(messages)) as stream:
                async for texto in stream.text_stream:
                    yield texto
        except Exception as e:
            raise LLMError(str(e), clasificar_error(e)) from e


class GeminiClient(BaseLLMClient):
    provider = Provider.GEMINI

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        super().__init__(api_key, model, temperature, max_tokens)
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))

    def _convertir_mensajes(self, messages: List[ChatMessage]):
        """Gemini separa el system prompt del resto, y llama 'model' al rol del asistente."""
        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                rol_gemini = "model" if m.role == "assistant" else "user"
                contents.append(types.Content(role=rol_gemini, parts=[types.Part(text=m.content)]))
        return contents, system_instruction

    def _config(self, system_instruction):
        return types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system_instruction,
        )

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            contents, system_instruction = self._convertir_mensajes(messages)
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._config(system_instruction),
            )
            return self._ok(response.text)
        except Exception as e:
            return self._error(e)

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            contents, system_instruction = self._convertir_mensajes(messages)
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self._config(system_instruction),
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(str(e), clasificar_error(e)) from e
