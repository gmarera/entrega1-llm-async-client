import asyncio
import logging
import os

from dotenv import load_dotenv
from pydantic import SecretStr

from manager import AsyncLLMManager
from schemas import ChatMessage, LLMConfig, Provider

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("google_genai").setLevel(logging.ERROR)  # sino tira un aviso de AFC en cada llamada


def leer_key(nombre: str):
    valor = os.getenv(nombre)
    return SecretStr(valor) if valor else None


config_openai = LLMConfig(
    provider=Provider.OPENAI,
    model="gpt-4o-mini",
    openai_api_key=leer_key("OPENAI_API_KEY"),
    max_tokens=1024,
)
config_anthropic = LLMConfig(
    provider=Provider.ANTHROPIC,
    model="claude-haiku-4-5-20251001",
    anthropic_api_key=leer_key("ANTHROPIC_API_KEY"),
    max_tokens=1024,
)
config_gemini = LLMConfig(
    provider=Provider.GEMINI,
    model="gemini-flash-latest",
    google_api_key=leer_key("GOOGLE_API_KEY"),
    max_tokens=1024,
)

pregunta = [ChatMessage(role="user", content="¿Qué es la entropía? Respondé en 2 líneas.")]


async def probar(nombre: str, config: LLMConfig):
    manager = AsyncLLMManager(config)

    print(f"\n--- {nombre}: modo normal ---")
    resultado = await manager.generate(pregunta)
    print(resultado.content if not resultado.error else f"❌ {resultado.error}")

    print(f"\n--- {nombre}: streaming ---")
    async for chunk in manager.generate_stream(pregunta):
        print(chunk, end="", flush=True)
    print()


async def probar_fallback():
    """OpenAI -> Anthropic -> Gemini. Si las dos primeras keys son inválidas responde Gemini."""
    manager = AsyncLLMManager(config_openai, fallbacks=[config_anthropic, config_gemini])

    print("\n--- Cadena con fallback (openai -> anthropic -> gemini) ---")
    resultado = await manager.generate(pregunta)
    if resultado.error:
        print(f"❌ {resultado.error}")
    else:
        print(f"Respondió: {resultado.provider.value} ({resultado.model})")
        print(resultado.content)

    print("\nLo mismo en streaming:")
    async for chunk in manager.generate_stream(pregunta):
        print(chunk, end="", flush=True)
    print()


async def main():
    await probar("🟢 OpenAI", config_openai)
    await probar("🟣 Anthropic", config_anthropic)
    await probar("🔵 Gemini", config_gemini)
    await probar_fallback()


if __name__ == "__main__":
    asyncio.run(main())