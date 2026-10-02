import pytest

from manager import AsyncLLMManager
from schemas import ErrorType, Provider
from tests.conftest import ClienteFalso, ClienteQueSeCorta, config


async def juntar(generador):
    return "".join([x async for x in generador])


# ---------- factory ----------
@pytest.mark.parametrize("provider", list(Provider))
def test_factory_crea_un_cliente_por_proveedor(provider):
    manager = AsyncLLMManager(config(provider))
    assert manager._clientes[0].provider == provider


def test_falta_la_api_key():
    cfg = config(Provider.OPENAI).model_copy(update={"openai_api_key": None})
    with pytest.raises(ValueError):
        AsyncLLMManager(cfg)


# ---------- modo normal ----------
async def test_camino_feliz(armar, pregunta):
    openai, gemini = ClienteFalso(Provider.OPENAI, ["hola!"]), ClienteFalso(Provider.GEMINI)
    resultado = await armar(openai, gemini).generate(pregunta)
    assert resultado.content == "hola!" and resultado.provider == Provider.OPENAI
    assert gemini.llamadas == 0


async def test_rate_limit_reintenta_y_despues_usa_el_fallback(armar, pregunta):
    openai = ClienteFalso(Provider.OPENAI, [ErrorType.RATE_LIMIT])  # siempre 429
    gemini = ClienteFalso(Provider.GEMINI, ["respondió gemini"])
    manager = armar(openai, gemini, max_reintentos=2)

    resultado = await manager.generate(pregunta)

    assert resultado.provider == Provider.GEMINI and resultado.content == "respondió gemini"
    assert openai.llamadas == 3  # 1 intento + 2 reintentos
    assert len(manager.esperas) == 2


async def test_si_se_recupera_no_hace_falta_el_fallback(armar, pregunta):
    openai = ClienteFalso(Provider.OPENAI, [ErrorType.RATE_LIMIT, "ya anduvo"])
    gemini = ClienteFalso(Provider.GEMINI)
    resultado = await armar(openai, gemini).generate(pregunta)
    assert resultado.provider == Provider.OPENAI and gemini.llamadas == 0


async def test_api_key_invalida_no_se_reintenta(armar, pregunta):
    openai = ClienteFalso(Provider.OPENAI, [ErrorType.AUTH])
    anthropic = ClienteFalso(Provider.ANTHROPIC, [ErrorType.AUTH])
    gemini = ClienteFalso(Provider.GEMINI, ["ok"])
    manager = armar(openai, anthropic, gemini)

    resultado = await manager.generate(pregunta)

    assert resultado.provider == Provider.GEMINI
    assert openai.llamadas == anthropic.llamadas == 1
    assert manager.esperas == []


async def test_backoff_exponencial(armar, pregunta):
    openai = ClienteFalso(Provider.OPENAI, [ErrorType.SERVER])
    manager = armar(openai, max_reintentos=3, espera_base=1.0)
    await manager.generate(pregunta)
    # 1s, 2s, 4s con un +-20% de azar
    for espera, esperado in zip(manager.esperas, [1, 2, 4]):
        assert esperado * 0.8 <= espera <= esperado * 1.2


async def test_si_fallan_todos_devuelve_error_sin_romper(armar, pregunta):
    manager = armar(ClienteFalso(Provider.OPENAI, [ErrorType.AUTH]),
                    ClienteFalso(Provider.GEMINI, [ErrorType.CONNECTION]))
    resultado = await manager.generate(pregunta)
    assert resultado.error and "openai" in resultado.error and "gemini" in resultado.error
    assert resultado.content == ""


async def test_circuit_breaker_deja_de_llamar_al_que_falla(armar, pregunta):
    openai = ClienteFalso(Provider.OPENAI, [ErrorType.AUTH])
    manager = armar(openai, ClienteFalso(Provider.GEMINI), fallos_para_pausar=2, segundos_pausa=999)
    for _ in range(5):
        await manager.generate(pregunta)
    assert openai.llamadas == 2  # a la segunda falla quedó pausado


# ---------- streaming ----------
async def test_stream_camino_feliz(armar, pregunta):
    manager = armar(ClienteFalso(Provider.OPENAI, ["uno dos tres"]))
    assert await juntar(manager.generate_stream(pregunta)) == "uno dos tres "


async def test_stream_fallback_si_falla_antes_del_primer_token(armar, pregunta):
    openai = ClienteFalso(Provider.OPENAI, [ErrorType.RATE_LIMIT])
    gemini = ClienteFalso(Provider.GEMINI, ["hola desde gemini"])
    texto = await juntar(armar(openai, gemini).generate_stream(pregunta))
    assert texto == "hola desde gemini "


async def test_stream_que_se_corta_a_la_mitad_no_cambia_de_proveedor(armar, pregunta):
    gemini = ClienteFalso(Provider.GEMINI, ["otra respuesta"])
    texto = await juntar(armar(ClienteQueSeCorta(Provider.OPENAI), gemini).generate_stream(pregunta))
    assert texto.startswith("parcial ") and "Error durante el streaming" in texto
    assert gemini.llamadas == 0


async def test_stream_si_fallan_todos_avisa_sin_romper(armar, pregunta):
    manager = armar(ClienteFalso(Provider.OPENAI, [ErrorType.AUTH]),
                    ClienteFalso(Provider.GEMINI, [ErrorType.AUTH]))
    texto = await juntar(manager.generate_stream(pregunta))
    assert "No se pudo generar" in texto
