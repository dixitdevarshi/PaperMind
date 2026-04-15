import sys
from src.logger import get_logger

logger = get_logger(__name__)


def _error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "unknown"
        line_number = -1

    error_message = (
        f"Error occurred in script: [{file_name}] "
        f"at line [{line_number}] — {str(error)}"
    )
    return error_message


class PaperMindException(Exception):
    """Base exception for all PaperMind errors."""

    def __init__(self, error_message: str | Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = _error_message_detail(
            error=Exception(str(error_message)),
            error_detail=error_detail,
        )
        logger.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message


class DocumentLoadError(PaperMindException):
    """Raised when a PDF or image cannot be loaded or parsed."""
    pass


class EmbeddingError(PaperMindException):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(PaperMindException):
    """Raised when ChromaDB operations fail."""
    pass


class RetrievalError(PaperMindException):
    """Raised when semantic search returns no usable results."""
    pass


class LLMError(PaperMindException):
    """Raised when the Claude API call fails."""
    pass


class RoutingError(PaperMindException):
    """Raised when the tool router cannot determine an appropriate tool."""
    pass


class EvaluationError(PaperMindException):
    """Raised when RAGAS evaluation pipeline fails."""
    pass


class ImageIngestionError(PaperMindException):
    """Raised when Claude Vision cannot extract content from an image."""
    pass