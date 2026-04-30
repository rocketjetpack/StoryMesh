"""OpenAI image generation client backed by the OpenAI Images API."""

from __future__ import annotations

import base64
import os

import openai

from storymesh.llm.image_base import GeneratedImage, ImageClient, register_image_provider

_DEFAULT_MODEL = "gpt-image-2"


class OpenAIImageClient(ImageClient):
    """ImageClient implementation backed by the OpenAI Images API (gpt-image-2).

    API key is resolved from the api_key argument or OPENAI_API_KEY env var.

    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        model: Model identifier. Defaults to 'gpt-image-2'.
        agent_name: Identifies the calling agent in log output.

    Raises:
        ValueError: If no API key is found in args or environment.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        agent_name: str = "unknown",
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key has not been provided. "
                "Pass api_key or set OPENAI_API_KEY in the environment."
            )
        resolved_model = model or _DEFAULT_MODEL
        super().__init__(model=resolved_model, agent_name=agent_name)
        self._api_key = resolved_key
        self.client = openai.OpenAI(api_key=resolved_key)

    def generate(
        self,
        prompt: str,
        *,
        size: str,
        quality: str,
    ) -> GeneratedImage:
        """Call the OpenAI Images API and return PNG bytes.

        gpt-image-2 always returns base64-encoded image data; no
        response_format parameter is required or accepted.

        Args:
            prompt: Text description of the desired image.
            size: Image dimensions. gpt-image-2 supports flexible sizes;
                common values are '1024x1024', '1024x1792', and '1792x1024'.
            quality: 'low', 'medium', 'high', or 'auto'.

        Returns:
            GeneratedImage with decoded PNG bytes. revised_prompt is always
            None as gpt-image-2 does not return a revised prompt.

        Raises:
            openai.OpenAIError: On API-level failures.
            ValueError: If the API response is missing expected fields.
        """
        response = self.client.images.generate(  # type: ignore[call-overload]
            model=self.model,
            prompt=prompt,
            n=1,
            size=size,
            quality=quality,
        )

        if not response.data:
            raise ValueError("OpenAI Images API returned empty data list.")

        image_data = response.data[0]
        if image_data.b64_json is None:
            raise ValueError("OpenAI Images API response missing b64_json field.")

        image_bytes = base64.b64decode(image_data.b64_json)
        revised_prompt: str | None = getattr(image_data, "revised_prompt", None)

        return GeneratedImage(image_bytes=image_bytes, revised_prompt=revised_prompt)


register_image_provider("openai", OpenAIImageClient)
