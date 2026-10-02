"""Los clientes reales con el SDK reemplazado por un mock (no se llama a ninguna API)."""
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from clients import AnthropicClient, GeminiClient, OpenAIClient, clasificar_error
from schemas import ChatMessage, ErrorType, Provider

MENSAJES = [ChatMessage(role="system", content="sé breve"), ChatMessage(role="user", content="hola")]


def respuesta_http(status):
    return httpx.Response(status, request=httpx.Request("POST", "https://ejemplo.test"))


def nuevo(clase):
    return clase(api_key="x", model="m", temperature=0.5, max_tokens=50)


async def test_openai_ok():
    c = nuevo(OpenAIClient)
    c._client = NS(chat=NS(completions=NS(create=AsyncMock(
        return_value=NS(choices=[NS(message=NS(content="hola!"))])))))
    r = await c.generate(MENSAJES)
    assert r.content == "hola!" and r.provider == Provider.OPENAI and r.error is None


async def test_openai_rate_limit_devuelve_error_en_vez_de_romper():
    c = nuevo(OpenAIClient)
    error = openai.RateLimitError("cuota", response=respuesta_http(429), body=None)
    c._client = NS(chat=NS(completions=NS(create=AsyncMock(side_effect=error))))
    r = await c.generate(MENSAJES)
    assert r.error_type == ErrorType.RATE_LIMIT and r.content == ""


async def test_respuesta_vacia_es_error():
    c = nuevo(OpenAIClient)
    c._client = NS(chat=NS(completions=NS(create=AsyncMock(
        return_value=NS(choices=[NS(message=NS(content=None))])))))
    assert (await c.generate(MENSAJES)).error is not None


async def test_anthropic_manda_el_system_aparte():
    c = nuevo(AnthropicClient)
    create = AsyncMock(return_value=NS(content=[NS(type="text", text="hola!")]))
    c._client = NS(messages=NS(create=create))
    r = await c.generate(MENSAJES)
    kwargs = create.await_args.kwargs
    assert r.content == "hola!"
    assert kwargs["system"] == "sé breve"
    assert all(m["role"] != "system" for m in kwargs["messages"])


def test_anthropic_solo_manda_parametros_que_el_sdk_acepta():
    # el AsyncMock acepta cualquier cosa, así que comparo contra la firma real del SDK
    import inspect

    from anthropic.resources.messages import AsyncMessages

    aceptados = set(inspect.signature(AsyncMessages.create).parameters)
    assert set(nuevo(AnthropicClient)._preparar(MENSAJES)) <= aceptados


async def test_anthropic_key_invalida():
    c = nuevo(AnthropicClient)
    error = anthropic.AuthenticationError("mala key", response=respuesta_http(401), body=None)
    c._client = NS(messages=NS(create=AsyncMock(side_effect=error)))
    assert (await c.generate(MENSAJES)).error_type == ErrorType.AUTH


async def test_gemini_ok_y_roles():
    c = nuevo(GeminiClient)
    generar = AsyncMock(return_value=NS(text="hola!"))
    c._client = NS(aio=NS(models=NS(generate_content=generar)))
    mensajes = MENSAJES + [ChatMessage(role="assistant", content="chau")]
    r = await c.generate(mensajes)
    kwargs = generar.await_args.kwargs
    assert r.content == "hola!" and r.provider == Provider.GEMINI
    assert [x.role for x in kwargs["contents"]] == ["user", "model"]
    assert kwargs["config"].system_instruction == "sé breve"


def test_clasificar_errores():
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://ejemplo.test"))
    sin_conexion = openai.APIConnectionError(request=httpx.Request("POST", "https://ejemplo.test"))
    gemini_429 = genai_errors.ClientError(429, {"error": {"message": "x", "status": "RESOURCE_EXHAUSTED"}})
    gemini_key = genai_errors.ClientError(400, {"error": {"message": "API key not valid.", "status": "INVALID_ARGUMENT"}})
    gemini_500 = genai_errors.ServerError(503, {"error": {"message": "x", "status": "UNAVAILABLE"}})
    assert clasificar_error(timeout) == ErrorType.TIMEOUT
    assert clasificar_error(sin_conexion) == ErrorType.CONNECTION
    assert clasificar_error(gemini_429) == ErrorType.RATE_LIMIT
    assert clasificar_error(gemini_key) == ErrorType.AUTH
    assert clasificar_error(gemini_500) == ErrorType.SERVER
    assert clasificar_error(ValueError("algo raro")) == ErrorType.OTHER


# ---------- test contra la API real de Gemini (opcional) ----------
@pytest.mark.skipif(not __import__("os").getenv("RUN_LIVE_TESTS"), reason="poné RUN_LIVE_TESTS=1 para correrlo")
async def test_gemini_real():
    import os

    from dotenv import load_dotenv

    load_dotenv()
    c = GeminiClient(api_key=os.environ["GOOGLE_API_KEY"], model="gemini-flash-latest",
                     temperature=0.2, max_tokens=200)
    r = await c.generate([ChatMessage(role="user", content="Respondé solo con la palabra: pong")])
    assert r.error is None and "pong" in r.content.lower()
