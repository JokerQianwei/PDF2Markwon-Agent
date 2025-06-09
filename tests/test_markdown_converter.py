import pytest
from pdf_to_markdown.markdown_converter import MarkdownConverter
from pdf_to_markdown.exceptions import MarkdownGenerationError

def test_to_markdown_with_multiple_pages():
    """
    Tests that OCR results from multiple pages are correctly merged
    with separators.
    """
    converter = MarkdownConverter()
    ocr_results = {
        1: "This is page one.",
        2: "This is page two.",
        3: "This is page three."
    }
    
    expected_markdown = (
        "This is page one.\n\n"
        "---\n\n"
        "This is page two.\n\n"
        "---\n\n"
        "This is page three."
    )
    
    assert converter.to_markdown(ocr_results) == expected_markdown

def test_to_markdown_with_single_page():
    """
    Tests that a single page is converted correctly without separators.
    """
    converter = MarkdownConverter()
    ocr_results = {1: "Hello, world."}
    assert converter.to_markdown(ocr_results) == "Hello, world."

def test_to_markdown_with_empty_results():
    """
    Tests that an empty dictionary of results produces an empty string.
    """
    converter = MarkdownConverter()
    ocr_results = {}
    assert converter.to_markdown(ocr_results) == ""

def test_save_to_file(mocker):
    """
    Tests that the save_to_file method correctly writes content to a file.
    """
    converter = MarkdownConverter()
    mock_file = mocker.mock_open()
    mocker.patch("builtins.open", mock_file)
    
    content = "My Markdown content."
    path = "test.md"
    
    converter.save_to_file(content, path)
    
    mock_file.assert_called_once_with(path, 'w', encoding='utf-8')
    mock_file().write.assert_called_once_with(content)

def test_save_to_file_io_error(mocker):
    """
    Tests that an IOError during file saving is caught and raises
    our custom exception.
    """
    converter = MarkdownConverter()
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("builtins.open", side_effect=IOError("Disk full"))

    with pytest.raises(MarkdownGenerationError, match="Disk full"):
        converter.save_to_file("some content", "test.md") 