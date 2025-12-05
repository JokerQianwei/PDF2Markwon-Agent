class PdfToMarkdownError(Exception):
    """Base exception for all custom errors in this application."""
    pass

class PDFConversionError(PdfToMarkdownError):
    """Raised when an error occurs during PDF to image conversion."""
    pass

class OCRProcessingError(PdfToMarkdownError):
    """Raised when an error occurs during the OCR process."""
    pass

class APIError(OCRProcessingError):
    """Raised for API-specific errors, e.g., authentication, rate limits."""
    pass

class MarkdownGenerationError(PdfToMarkdownError):
    """Raised when an error occurs during Markdown generation."""
    pass

class ConfigurationError(PdfToMarkdownError):
    """Raised for configuration-related errors."""
    pass 