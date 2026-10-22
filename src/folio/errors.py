"""Expected failures raised by Folio.

Callers (CLI, API, UI) catch :class:`FolioError` and show ``str(exc)`` to the
user. Programming errors are left as built-in exceptions.
"""

from __future__ import annotations


class FolioError(Exception):
    """Base class for errors the user can act on."""


class IngestError(FolioError):
    """A file could not be read or contained no usable text."""


class ProviderError(FolioError):
    """An embedding model, language model, or vector store is unavailable."""


class ConfigError(FolioError):
    """Configuration is missing or contradictory."""
