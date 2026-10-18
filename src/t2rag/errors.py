"""Project-specific exceptions.

Using named exceptions lets the command-line interface show useful messages
without hiding unexpected programming errors.
"""


class T2RagError(Exception):
    """Base class for expected project errors."""


class ConfigurationError(T2RagError):
    """Raised when a required setting is missing or invalid."""


class LeakageError(T2RagError):
    """Raised when forbidden question or answer data reaches an index path."""


class TestSplitSealedError(T2RagError):
    """Raised when a development command tries to read sealed test labels."""


class ExternalServiceError(T2RagError):
    """Raised when Ollama or DeepInfra returns an unusable response."""
