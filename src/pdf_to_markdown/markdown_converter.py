from .logger import logger
from .exceptions import MarkdownGenerationError

class MarkdownConverter:
    """Converts a collection of OCR text results into a single Markdown document."""

    def to_markdown(self, ocr_results: dict[int, str]) -> str:
        """
        Merges OCR results from multiple pages into a single Markdown string.

        Args:
            ocr_results (dict[int, str]): A dictionary mapping page numbers
                                         to their OCR text.

        Returns:
            str: A single string containing the combined Markdown content.
        """
        full_markdown = []
        
        # Sort by page number to ensure correct order
        sorted_pages = sorted(ocr_results.keys())
        
        for page_number in sorted_pages:
            page_content = ocr_results.get(page_number, "")
            full_markdown.append(page_content)
        
        # Join pages with a separator to indicate page breaks
        # Using a double newline, a horizontal rule, and another double newline
        # for clear separation.
        return "\n\n---\n\n".join(full_markdown)

    def save_to_file(self, markdown_content: str, output_path: str):
        """
        Saves the given Markdown content to a file.

        Args:
            markdown_content (str): The Markdown string to save.
            output_path (str): The path to the output file.
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.info(f"Successfully saved Markdown to {output_path}")
        except IOError as e:
            msg = f"Error saving file to {output_path}: {e}"
            logger.error(msg)
            raise MarkdownGenerationError(msg) from e 