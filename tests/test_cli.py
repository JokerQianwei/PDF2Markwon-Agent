import pytest
from click.testing import CliRunner
from pdf_to_markdown.cli import main
from pdf_to_markdown.exceptions import PdfToMarkdownError

def test_cli_success(mocker):
    """Tests the CLI for a successful run."""
    # Mock all the components in the processing chain
    mocker.patch('pdf_to_markdown.cli.PDFConverter.convert_to_images', return_value=['fake_image.png'])
    mock_ocr_client = mocker.patch('pdf_to_markdown.cli.OcrClient')
    mocker.patch('pdf_to_markdown.cli.OCRProcessor.process_images', return_value={1: 'ocr text'})
    mocker.patch('pdf_to_markdown.cli.MarkdownConverter.to_markdown', return_value='final markdown')
    mock_save = mocker.patch('pdf_to_markdown.cli.MarkdownConverter.save_to_file')

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("test.pdf", "w") as f:
            f.write("dummy pdf")

        result = runner.invoke(main, ['--input-path', 'test.pdf', '--output-path', 'out.md', '--model', 'test-model'])

        assert result.exit_code == 0
        mock_ocr_client.assert_called_once_with(model='test-model')
        mock_save.assert_called_once_with('final markdown', 'out.md')

def test_cli_pdf_conversion_fails(mocker):
    """Tests the CLI when PDF conversion fails."""
    mocker.patch(
        'pdf_to_markdown.cli.PDFConverter.convert_to_images',
        side_effect=PdfToMarkdownError("PDF is corrupted")
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("test.pdf", "w") as f:
            f.write("dummy pdf")

        result = runner.invoke(main, ['--input-path', 'test.pdf'])

        assert result.exit_code == 0 # The CLI should exit gracefully
        assert "Operation failed: PDF is corrupted" in result.output

def test_cli_api_key_error(mocker):
    """Tests the CLI when no API key is configured."""
    mocker.patch('pdf_to_markdown.cli.PDFConverter.convert_to_images', return_value=['fake_image.png'])
    mocker.patch('pdf_to_markdown.cli.OcrClient', side_effect=PdfToMarkdownError("No API key"))

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("test.pdf", "w") as f:
            f.write("dummy pdf")

        result = runner.invoke(main, ['--input-path', 'test.pdf'])

        assert result.exit_code == 0
        assert "Operation failed: No API key" in result.output

def test_cli_verbose_logging(mocker):
    """Tests that the verbose flag sets the log level to DEBUG."""
    mock_setup_logger = mocker.patch('pdf_to_markdown.cli.setup_logger')
    mocker.patch('pdf_to_markdown.cli.PDFConverter')
    mock_ocr_client_init = mocker.patch('pdf_to_markdown.cli.OcrClient')
    mocker.patch('pdf_to_markdown.cli.OCRProcessor')
    mocker.patch('pdf_to_markdown.cli.MarkdownConverter')

    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("test.pdf", "w") as f:
            f.write("dummy pdf")

        runner.invoke(main, ['--input-path', 'test.pdf', '--verbose'])

        # Check that setup_logger was called with logging.DEBUG
        import logging
        mock_setup_logger.assert_called_once_with(logging.DEBUG)
        mock_ocr_client_init.assert_called_once_with(model=None) 