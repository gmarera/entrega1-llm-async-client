import pytest
from pydantic import SecretStr

from clients import BaseLLMClient
from manager import AsyncLLMManager
from schemas import ChatMessage, ErrorType, LLMConfig, LLMError, ModelResponse, Provider


class ClienteFalso(BaseLLMClient):
    """Cliente que responde lo que le programemos: un texto (ok) o un ErrorType (falla)."""

    def __init__(self, provider, guion=("ok",)):
        self.provider = provider
        self.model = "falso"
        self.guion = list(guion)
        self.llamadas = 0

    def _siguiente(self):
        item = self.guion[min(self.llamadas, len(self.guion) - 1)]
        self.llamadas += 1
        return item

    async def generate(self, messages):
        item = self._siguiente()
        if isinstance(item, ErrorType):
            return ModelResponse(provider=self.provider, model=self.model, content="",
                                 error=f"[{item.value}] error simulado", error_type=item)
        return ModelResponse(provider=self.provider, model=self.model, content=item)

    async def generate_stream(self, messages):
        item = self._siguiente()
        if isinstance(item, ErrorType):
            raise LLMError("error simulado", item)
        for palabra in item.split():
            yield palabra + " "


class ClienteQueSeCorta(ClienteFalso):
    """Manda un token y después falla."""

    async def generate_stream(self, messages):
        self.llamadas += 1
        yield "parcial "
        raise LLMError("se cortó", ErrorType.CONNECTION)


def config(provider):
    return LLMConfig(
        provider=provider, model="x",
        openai_api_key=SecretStr("a"), anthropic_api_key=SecretStr("b"), google_api_key=SecretStr("c"),
    )


@pytest.fixture
def pregunta():
    return [ChatMessage(role="user", content="hola")]


@pytest.fixture
def armar():
    """armar(c1, c2, ...) -> manager que usa esos clientes falsos, sin esperas reales."""
    esperas = []

    async def no_esperar(segundos):
        esperas.append(segundos)

    def _armar(*clientes, **kwargs):
        manager = AsyncLLMManager(config(Provider.OPENAI), esperar=no_esperar, **kwargs)
        manager._clientes = list(clientes)
        manager.esperas = esperas
        return manager

    return _armar
