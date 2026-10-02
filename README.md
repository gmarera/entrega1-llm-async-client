# Unified Async LLM Client

Pre-entrega 1 del curso AI Engineering (Coderhouse). Es un cliente asíncrono para hablar con
varios LLMs (OpenAI, Anthropic y Gemini) usando siempre la misma interfaz, con streaming,
validación con Pydantic y manejo de errores.

La idea es poder cambiar de proveedor tocando solo la configuración, sin modificar el resto del código.
Si un proveedor falla (rate limit, caída de red, key inválida) el programa no se rompe: reintenta y,
si sigue fallando, pasa al siguiente.

## Archivos

- `schemas.py`: modelos de Pydantic (`ChatMessage`, `LLMConfig`, `ModelResponse`) y los tipos de error.
- `clients.py`: la clase abstracta `BaseLLMClient` y los tres clientes (`OpenAIClient`, `AnthropicClient`, `GeminiClient`).
- `manager.py`: `AsyncLLMManager`. Crea el cliente según el proveedor de la config (Factory) y se encarga de reintentos, fallback y circuit breaker.
- `main.py`: script de prueba (modo normal y streaming con "¿Qué es la entropía?").
- `tests/`: tests con pytest.

## Cómo correrlo

Hace falta Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # y completar las keys
python main.py
```

### Variables de entorno

| Variable | Para qué |
|---|---|
| `OPENAI_API_KEY` | key de OpenAI |
| `ANTHROPIC_API_KEY` | key de Anthropic |
| `GOOGLE_API_KEY` | key de Gemini (se saca gratis en https://aistudio.google.com/apikey) |

No hace falta que las tres sean válidas. En mi caso solo la de Gemini es real, las otras dos tienen
un valor de mentira a propósito. Así se ve cómo `main.py` muestra el error de cada proveedor sin
romperse y cómo la cadena con fallback termina respondiendo con Gemini.

## Qué hace `main.py`

1. Prueba cada proveedor por separado, en modo normal y en streaming.
2. Arma una cadena `openai -> anthropic -> gemini` y hace la misma pregunta. Imprime qué proveedor terminó respondiendo.

Los avisos de reintentos y fallbacks salen por `logging` (nivel WARNING).

## Cómo se maneja la resiliencia

- Los errores se clasifican en `rate_limit`, `timeout`, `connection`, `server`, `auth` y `other`.
- Los primeros cuatro son pasajeros, así que se reintenta con backoff exponencial (1s, 2s, 4s...) hasta `max_reintentos` veces.
- Una key inválida (`auth`) no se reintenta, se pasa directo al siguiente proveedor.
- Si un proveedor falla 3 veces seguidas queda pausado 30 segundos (circuit breaker simple).
- Si fallan todos, `generate()` devuelve un `ModelResponse` con el campo `error` completo, no tira excepción.
- En streaming solo se puede cambiar de proveedor antes de que llegue el primer token. Si se corta a la mitad se avisa con un mensaje al final del texto, para no mezclar respuestas de dos modelos.

Los SDKs se crean con `max_retries=0` para que los reintentos los haga el manager y no se junten con los del SDK.

## Uso rápido

```python
from pydantic import SecretStr
from manager import AsyncLLMManager
from schemas import ChatMessage, LLMConfig, Provider

config = LLMConfig(
    provider=Provider.GEMINI,
    model="gemini-flash-latest",
    google_api_key=SecretStr("..."),
)
manager = AsyncLLMManager(config)

respuesta = await manager.generate([ChatMessage(role="user", content="Hola")])
print(respuesta.content or respuesta.error)

async for chunk in manager.generate_stream([ChatMessage(role="user", content="Hola")]):
    print(chunk, end="", flush=True)
```

Para usar fallback se pasa una lista de configs de respaldo:
`AsyncLLMManager(config_openai, fallbacks=[config_anthropic, config_gemini])`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

No usan internet ni API keys: los SDKs se reemplazan por mocks. Prueban la validación de Pydantic,
la factory, los reintentos con backoff, el fallback cuando OpenAI devuelve un rate limit (simulado),
el caso de key inválida, cuando fallan todos los proveedores y el streaming con fallback.

Hay un test extra que llama a Gemini de verdad, solo corre si se define `RUN_LIVE_TESTS=1`
(y hay una `GOOGLE_API_KEY` válida en el `.env`).

## Notas

- Modelos usados: `gpt-4o-mini`, `claude-haiku-4-5-20251001` y `gemini-flash-latest`. Se cambian desde `main.py`.
- A Anthropic no se le manda `temperature`: la versión actual del SDK no la acepta en `messages.create`.
- El `.env` está en el `.gitignore`, nunca hay que subirlo.
