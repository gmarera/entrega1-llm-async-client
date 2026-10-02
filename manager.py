import asyncio
import logging
import random
import time
from typing import AsyncGenerator, List, Optional

from clients import AnthropicClient, BaseLLMClient, GeminiClient, OpenAIClient
from schemas import (
    ChatMessage,
    ERRORES_REINTENTABLES,
    ErrorType,
    LLMConfig,
    LLMError,
    ModelResponse,
    Provider,
)

logger = logging.getLogger(__name__)

# para cada proveedor: la clase del cliente y el campo de la config donde está su api key
CLIENTES = {
    Provider.OPENAI: (OpenAIClient, "openai_api_key"),
    Provider.ANTHROPIC: (AnthropicClient, "anthropic_api_key"),
    Provider.GEMINI: (GeminiClient, "google_api_key"),
}


class AsyncLLMManager:
    """Factory + resiliencia.

    Se crea con la config del proveedor principal y, si se quiere, una lista de configs de
    respaldo. Si el principal falla (después de reintentar) pasa al siguiente.
    """

    def __init__(
        self,
        config: LLMConfig,
        fallbacks: Optional[List[LLMConfig]] = None,
        max_reintentos: int = 2,
        espera_base: float = 1.0,
        max_espera: float = 8.0,
        fallos_para_pausar: int = 3,
        segundos_pausa: float = 30.0,
        esperar=asyncio.sleep,  # se puede reemplazar en los tests para no esperar de verdad
    ):
        self.config = config
        self._clientes: List[BaseLLMClient] = [
            self._crear_cliente(c) for c in [config, *(fallbacks or [])]
        ]
        self.max_reintentos = max_reintentos
        self.espera_base = espera_base
        self.max_espera = max_espera
        self.fallos_para_pausar = fallos_para_pausar
        self.segundos_pausa = segundos_pausa
        self._esperar = esperar
        # circuit breaker simple: si un proveedor falla N veces seguidas lo dejamos descansar un rato
        self._fallos_seguidos = {}
        self._pausado_hasta = {}

    # ---------- Factory ----------
    def _crear_cliente(self, config: LLMConfig) -> BaseLLMClient:
        if config.provider not in CLIENTES:
            raise ValueError(f"Proveedor no soportado: {config.provider}")
        clase, campo_key = CLIENTES[config.provider]
        api_key = getattr(config, campo_key)
        if not api_key:
            raise ValueError(f"Falta {campo_key} en la configuración")
        return clase(
            api_key=api_key.get_secret_value(),
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    # ---------- circuit breaker ----------
    def _pausado(self, cliente: BaseLLMClient) -> bool:
        hasta = self._pausado_hasta.get(cliente.provider)
        return hasta is not None and time.monotonic() < hasta

    def _anotar_fallo(self, cliente: BaseLLMClient):
        p = cliente.provider
        self._fallos_seguidos[p] = self._fallos_seguidos.get(p, 0) + 1
        if self._fallos_seguidos[p] >= self.fallos_para_pausar:
            self._pausado_hasta[p] = time.monotonic() + self.segundos_pausa
            logger.warning("%s falló %s veces seguidas, lo pauso %ss",
                           p.value, self._fallos_seguidos[p], self.segundos_pausa)

    def _anotar_ok(self, cliente: BaseLLMClient):
        self._fallos_seguidos[cliente.provider] = 0
        self._pausado_hasta.pop(cliente.provider, None)

    # ---------- retry con backoff exponencial ----------
    def _tiempo_de_espera(self, intento: int) -> float:
        espera = min(self.max_espera, self.espera_base * 2**intento)  # 1s, 2s, 4s...
        return espera * random.uniform(0.8, 1.2)  # un poquito de azar para no reintentar todos juntos

    async def _generar_con_reintentos(self, cliente, messages) -> ModelResponse:
        for intento in range(self.max_reintentos + 1):
            respuesta = await cliente.generate(messages)
            if not respuesta.error or respuesta.error_type not in ERRORES_REINTENTABLES:
                return respuesta
            if intento < self.max_reintentos:
                espera = self._tiempo_de_espera(intento)
                logger.warning("%s: %s -> reintento %s/%s en %.1fs", cliente.provider.value,
                               respuesta.error, intento + 1, self.max_reintentos, espera)
                await self._esperar(espera)
        return respuesta

    # ---------- modo normal ----------
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        errores = []
        for cliente in self._clientes:
            if self._pausado(cliente):
                errores.append(f"{cliente.provider.value}: pausado por fallos anteriores")
                continue
            respuesta = await self._generar_con_reintentos(cliente, messages)
            if not respuesta.error:
                self._anotar_ok(cliente)
                return respuesta
            self._anotar_fallo(cliente)
            errores.append(f"{cliente.provider.value}: {respuesta.error}")
            logger.warning("%s falló, paso al siguiente proveedor", cliente.provider.value)

        # si llegamos acá fallaron todos: devolvemos el error, no tiramos excepción
        return ModelResponse(
            provider=self.config.provider,
            model=self.config.model,
            content="",
            error="Fallaron todos los proveedores -> " + " | ".join(errores),
            error_type=ErrorType.OTHER,
        )

    # ---------- streaming ----------
    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        errores = []
        for cliente in self._clientes:
            if self._pausado(cliente):
                errores.append(f"{cliente.provider.value}: pausado por fallos anteriores")
                continue

            for intento in range(self.max_reintentos + 1):
                hubo_tokens = False
                try:
                    async for chunk in cliente.generate_stream(messages):
                        hubo_tokens = True
                        yield chunk
                    if hubo_tokens:
                        self._anotar_ok(cliente)
                        return
                    error = LLMError("el stream vino vacío")
                except LLMError as e:
                    if hubo_tokens:
                        # ya mandamos texto, no se puede cambiar de proveedor a la mitad
                        self._anotar_fallo(cliente)
                        yield f"\n[⚠️ Error durante el streaming: {e}]"
                        return
                    error = e

                if error.error_type not in ERRORES_REINTENTABLES:
                    break
                if intento < self.max_reintentos:
                    espera = self._tiempo_de_espera(intento)
                    logger.warning("%s: %s -> reintento %s/%s en %.1fs", cliente.provider.value,
                                   error, intento + 1, self.max_reintentos, espera)
                    await self._esperar(espera)

            self._anotar_fallo(cliente)
            errores.append(f"{cliente.provider.value}: {error}")
            logger.warning("%s falló, paso al siguiente proveedor", cliente.provider.value)

        yield "[⚠️ No se pudo generar la respuesta -> " + " | ".join(errores) + "]"
