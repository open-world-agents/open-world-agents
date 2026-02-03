"""
Immutable configuration for image token handling.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageTokenConfig:
    """Immutable configuration for image token handling in event tokenization.

    This config defines how image placeholders from EventEncoder are converted
    to actual image tokens that the MLLM understands.

    Attributes:
        prefix: Token(s) before the repeated image token (e.g., "<img>")
        token: The image token to repeat (e.g., "<IMG_CONTEXT>")
        length: Number of times to repeat the image token
        suffix: Token(s) after the repeated image token (e.g., "</img>")
        fake_placeholder: Internal placeholder used by EventEncoder (not in vocab)

    Example:
        For InternVL3:
            prefix="<img>", token="<IMG_CONTEXT>", length=256, suffix="</img>"
            Result: "<img><IMG_CONTEXT><IMG_CONTEXT>...(256 times)...</img>"

        For SmolVLM2:
            prefix="<fake_token_around_image><global-img>", token="<image>", length=64, suffix="<fake_token_around_image>"
    """

    prefix: str
    token: str
    length: int
    suffix: str
    fake_placeholder: str = "<fake_image_placeholder>"

    @property
    def pattern(self) -> str:
        """Generate the full image token pattern for replacement."""
        return f"{self.prefix}{self.token * self.length}{self.suffix}"

    @property
    def vocab_tokens(self) -> set[str]:
        """Get the set of tokens that need to be added to tokenizer vocab.

        Note: fake_placeholder is NOT included as it's not a real token in the vocab.
        It's an internal placeholder used by EventEncoder that gets replaced with
        the actual image token pattern during tokenization.
        """
        # TODO: prefix/suffix can be composed of multiple tokens, may need parsing
        return {self.prefix, self.token, self.suffix}
