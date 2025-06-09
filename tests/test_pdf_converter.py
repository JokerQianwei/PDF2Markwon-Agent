import pytest
import os
import fitz  # PyMuPDF
from pdf_to_markdown.pdf_converter import PDFConverter
from pdf_to_markdown.exceptions import PDFConversionError

@pytest.fixture(scope="session")
def sample_pdf(tmpdir_factory):
    """Creates a simple one-page PDF file for testing."""
    # Use a session-scoped temp directory to create the PDF only once
    fn = tmpdir_factory.mktemp("data").join("test.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Hello, world!")
    doc.save(str(fn))
    doc.close()
    return str(fn)

def test_convert_to_images_success(sample_pdf, tmpdir):
    """
    Tests successful conversion of a PDF to images.
    """
    converter = PDFConverter()
    output_dir = str(tmpdir)
    image_paths = converter.convert_to_images(sample_pdf, output_dir)

    assert len(image_paths) == 1
    assert os.path.exists(image_paths[0])
    assert image_paths[0].endswith("page_0001.png")

def test_convert_non_existent_pdf_raises_error():
    """
    Tests that an appropriate exception is raised for a non-existent PDF.
    """
    converter = PDFConverter()
    with pytest.raises(PDFConversionError, match="no such file"):
        converter.convert_to_images("non_existent.pdf", "any_dir")

def test_convert_corrupted_pdf_raises_error(mocker, tmpdir):
    """
    Tests that an appropriate exception is raised for a corrupted PDF.
    We simulate this by making fitz.open raise a generic error.
    """
    mocker.patch("fitz.open", side_effect=RuntimeError("corrupted file"))
    converter = PDFConverter()
    
    with pytest.raises(PDFConversionError, match="corrupted file"):
        converter.convert_to_images("dummy.pdf", str(tmpdir)) 