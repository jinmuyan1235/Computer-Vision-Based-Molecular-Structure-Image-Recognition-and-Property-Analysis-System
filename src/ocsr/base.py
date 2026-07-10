"""Common interfaces for interchangeable OCSR backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass
class OCSRResult:
    """Normalized output returned by every OCSR adapter."""

    smiles: str | None
    confidence: float | None
    backend: str
    status: Literal["success", "failed"]
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert this result to a JSON-serializable dictionary."""
        return asdict(self)


class BaseOCSRAdapter(ABC):
    """Abstract interface implemented by all recognition adapters."""

    backend_name = "base"

    @abstractmethod
    def recognize(self, image_path_or_array: Any) -> OCSRResult:
        """Recognize a molecular image and return an OCSRResult."""
        raise NotImplementedError
