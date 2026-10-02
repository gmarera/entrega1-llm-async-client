from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, SecretStr, field_validator


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ErrorType(str, Enum):
    """Para saber si vale la pena reintentar o si hay que pasar directo al fallback."""
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVER = "server"
    AUTH = "auth"
    OTHER = "other"


# estos son errores "pasajeros": reintentar tiene sentido. Una API key mala no se arregla reintentando
ERRORES_REINTENTABLES = {
    ErrorType.RATE_LIMIT,
    ErrorType.TIMEOUT,
    ErrorType.CONNECTION,
    ErrorType.SERVER,
}


class ChatMessage(BaseModel):
    role: str = Field(description="'user', 'assistant' o 'system'")
    content: str = Field(min_length=1)

    @field_validator("role")
    @classmethod
    def rol_valido(cls, v: str) -> str:
        roles_permitidos = {"user", "assistant", "system"}
        if v not in roles_permitidos:
            raise ValueError(f"role debe ser uno de {roles_permitidos}, recibido: '{v}'")
        return v


class LLMConfig(BaseModel):
    provider: Provider
    model: str
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    google_api_key: Optional[SecretStr] = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)


class ModelResponse(BaseModel):
    provider: Provider
    model: str
    content: str
    error: Optional[str] = None
    error_type: Optional[ErrorType] = None


class LLMError(Exception):
    """La usan los clientes en modo streaming (un generador no puede 'devolver' un error)."""

    def __init__(self, mensaje: str, error_type: ErrorType = ErrorType.OTHER):
        super().__init__(mensaje)
        self.error_type = error_type
