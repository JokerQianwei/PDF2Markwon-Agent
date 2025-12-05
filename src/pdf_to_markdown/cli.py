import click
import os
import tempfile
import shutil
import logging
from dotenv import load_dotenv

from .pdf_converter import PDFConverter
from .ocr_client import OcrClient
from .ocr_processor import OCRProcessor
from .markdown_converter import MarkdownConverter
from .logger import setup_logger
from .exceptions import PdfToMarkdownError

@click.command()
@click.option(
    '--input-path', '-i',
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help='The path to the input PDF file.'
)
@click.option(
    '--output-path', '-o',
    type=click.Path(dir_okay=False, writable=True),
    help='The path for the output Markdown file. Defaults to the same name as the input file.'
)
@click.option(
    '--temp-dir',
    type=click.Path(file_okay=False, writable=True),
    help='Directory for temporary image files. A temporary directory is created and cleaned up if not provided.'
)
@click.option(
    '--model', '-m',
    help="The OCR model to use (e.g., 'google/gemini-pro')."
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help="Enable verbose logging."
)
def main(input_path, output_path, temp_dir, model, verbose):
    """
    A command-line tool to convert PDF documents into Markdown files using OCR.
    """
    load_dotenv()
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logger(log_level)

    if not output_path:
        # Default output path is the same as input but with .md extension
        base, _ = os.path.splitext(input_path)
        output_path = base + '.md'

    # Determine if we need to manage a temporary directory
    use_managed_temp_dir = not temp_dir
    if use_managed_temp_dir:
        # Create a temporary directory that will be automatically cleaned up
        temp_dir = tempfile.mkdtemp()
    
    logger.info(f"Processing {input_path}...")
    
    try:
        # 1. Convert PDF to Images
        pdf_converter = PDFConverter()
        image_paths = pdf_converter.convert_to_images(input_path, temp_dir)

        if not image_paths:
            logger.error("Failed to convert PDF to images. Aborting.")
            click.echo("Failed to convert PDF to images. Aborting.", err=True)
            return

        # 2. Process images with OCR
        ocr_client = OcrClient(model=model)
        ocr_processor = OCRProcessor(ocr_client)
        ocr_results = ocr_processor.process_images(image_paths)

        # 3. Convert OCR results to Markdown
        markdown_converter = MarkdownConverter()
        markdown_content = markdown_converter.to_markdown(ocr_results)
        
        # 4. Save to file
        markdown_converter.save_to_file(markdown_content, output_path)
        
        logger.info(f"Successfully converted PDF to Markdown at {output_path}")

    except PdfToMarkdownError as e:
        logger.error(f"A controlled error occurred: {e}", exc_info=verbose)
        # exc_info=verbose will show stack trace only in verbose mode
        click.echo(f"Operation failed: {e}", err=True)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        click.echo(f"An unexpected error occurred. Check logs for details.", err=True)

    finally:
        # Clean up the temporary directory if we created it
        if use_managed_temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == '__main__':
    main() 