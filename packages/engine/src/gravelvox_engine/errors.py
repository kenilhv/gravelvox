"""Exceptions raised by the engine. The CLI maps each one to an exit code."""


class GravelvoxError(Exception):
    """Base class for engine errors."""


class DependencyError(GravelvoxError):
    """A runtime dependency is missing or failed to load."""


class MissingDependencyError(DependencyError):
    """A runtime dependency (such as faster-whisper) is not installed."""


class InvalidOptionError(GravelvoxError):
    """An option has a value the engine cannot use, such as an unknown language code."""


class ModelLoadError(GravelvoxError):
    """The model could not be found, downloaded or loaded."""


class AudioDecodeError(GravelvoxError):
    """The audio file could not be opened or decoded, or it holds no samples."""


class TranscriptionError(GravelvoxError):
    """The model failed while transcribing, for example because GPU libraries are missing."""
