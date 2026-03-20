from enum import Enum
from pydantic import BaseModel, Field


class AllowedFileType(str, Enum):
    PDF = "pdf"
    TXT = "txt"


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: dict = Field(default_factory=dict)


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"
