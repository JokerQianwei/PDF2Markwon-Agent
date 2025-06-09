import pytest
from unittest.mock import MagicMock
from pdf_to_markdown.ocr_processor import OCRProcessor

@pytest.fixture
def mock_ocr_client():
    """Fixture to create a mock OcrClient."""
    return MagicMock()

def test_process_images_success(mock_ocr_client):
    """Tests successful processing of multiple images."""
    mock_ocr_client.process_image_for_ocr.return_value = "Mocked OCR text"
    processor = OCRProcessor(mock_ocr_client)
    
    image_paths = ["path/to/image1.png", "path/to/image2.png"]
    results = processor.process_images(image_paths)

    assert len(results) == 2
    assert results[1] == "Mocked OCR text"
    assert results[2] == "Mocked OCR text"
    assert mock_ocr_client.process_image_for_ocr.call_count == 2

def test_process_images_with_retries(mock_ocr_client, mocker):
    """Tests the retry logic when the OCR call fails initially."""
    mocker.patch('time.sleep', return_value=None) # To speed up the test
    
    # Fail twice, then succeed
    mock_ocr_client.process_image_for_ocr.side_effect = [
        "", 
        "", 
        "Successful OCR text"
    ]
    processor = OCRProcessor(mock_ocr_client)
    
    image_paths = ["path/to/image1.png"]
    results = processor.process_images(image_paths, max_retries=3)

    assert len(results) == 1
    assert results[1] == "Successful OCR text"
    assert mock_ocr_client.process_image_for_ocr.call_count == 3

def test_process_images_full_failure(mock_ocr_client, mocker):
    """Tests when the OCR call fails for all retries."""
    mocker.patch('time.sleep', return_value=None)
    
    # Always return an empty string to simulate persistent failure
    mock_ocr_client.process_image_for_ocr.return_value = ""
    processor = OCRProcessor(mock_ocr_client)
    
    image_paths = ["path/to/image1.png"]
    results = processor.process_images(image_paths, max_retries=2)

    assert len(results) == 1
    assert results[1] == "" # Should contain an empty string on failure
    assert mock_ocr_client.process_image_for_ocr.call_count == 2 