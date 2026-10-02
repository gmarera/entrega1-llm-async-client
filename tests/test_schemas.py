import pytest
from pydantic import ValidationError

from schemas import ChatMessage, LLMConfig, Provider


def test_temperatura_fuera_de_rango():
    with pytest.raises(ValidationError):
        LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", temperature=5)


def test_max_tokens_tiene_que_ser_positivo():
    with pytest.raises(ValidationError):
        LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", max_tokens=0)


def test_rol_invalido():
    with pytest.raises(ValidationError):
        ChatMessage(role="robot", content="hola")


def test_mensaje_vacio():
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")


def test_la_api_key_no_se_imprime():
    cfg = LLMConfig(provider=Provider.OPENAI, model="m", openai_api_key="sk-secreta")
    assert "sk-secreta" not in repr(cfg)
