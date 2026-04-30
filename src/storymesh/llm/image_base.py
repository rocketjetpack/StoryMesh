"""Abstract base class and registry for image generation clients."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod


@dataclasses.dataclass(frozen=True)
class GeneratedImage:
    """Raw result from an image generation API call.

    image_bytes is raw PNG data ready to write to disk.
    revised_prompt is the provider-rewritten prompt, if any.
    """

    image_bytes: bytes
    revised_prompt: str | None


class ImageClient(ABC):
    """Vendor-agnostic interface for image generation.

    Subclasses implement generate() for a specific provider.
    """

    def __init__(self, *, model: str, agent_name: str = "unknown") -> None:
        self.model = model
        self.agent_name = agent_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        size: str,
        quality: str,
    ) -> GeneratedImage:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            size: Image dimensions (e.g. '1024x1792').
            quality: Quality tier (e.g. 'auto', 'low', 'medium', 'high').

        Returns:
            GeneratedImage with raw PNG bytes and optional revised prompt.
        """
        ...


# ---------------------------------------------------------------------------
# Provider registry — mirrors storymesh.llm.base
# ---------------------------------------------------------------------------

_IMAGE_REGISTRY: dict[str, type[ImageClient]] = {}


def register_image_provider(name: str, cls: type[ImageClient]) -> None:
    """Register an ImageClient subclass for a provider name.

    Idempotent for the same class; raises if a different class is registered
    under an already-taken name.

    Args:
        name: Provider name string (e.g. ``'openai'``).
        cls: The concrete ImageClient subclass to instantiate for this provider.

    Raises:
        ValueError: If ``name`` is already registered to a different class.
    """
    if name in _IMAGE_REGISTRY and _IMAGE_REGISTRY[name] is not cls:
        raise ValueError(
            f"Image provider '{name}' is already registered to "
            f"{_IMAGE_REGISTRY[name].__name__}, cannot re-register to {cls.__name__}."
        )
    _IMAGE_REGISTRY[name] = cls


def get_image_provider_class(name: str) -> type[ImageClient]:
    """Return the ImageClient subclass registered for the given provider name.

    Args:
        name: Provider name string.

    Returns:
        The registered ImageClient subclass.

    Raises:
        ValueError: If no provider is registered under ``name``.
    """
    if name not in _IMAGE_REGISTRY:
        registered = ", ".join(sorted(_IMAGE_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"Unknown image provider: '{name}'. Registered providers: {registered}"
        )
    return _IMAGE_REGISTRY[name]
