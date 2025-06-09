import pytest
from unittest.mock import MagicMock
from pdf_to_markdown.ocr_client import OcrClient
from pdf_to_markdown.exceptions import ConfigurationError, APIError

def test_client_initialization_priority(mocker):
    """Tests that OpenRouter client is prioritized when both keys are present."""
    mocker.patch.dict('os.environ', {
        "OPENROUTER_API_KEY": "fake_or_key",
        "GOOGLE_API_KEY": "fake_google_key"
    })
    mock_openai = mocker.patch('openai.OpenAI')
    client = OcrClient()
    assert client.client_type == "openrouter"
    mock_openai.assert_called_once_with(
        base_url="https://openrouter.ai/api/v1",
        api_key="fake_or_key",
    )

def test_client_fallback_to_google(mocker):
    """Tests that Google Gemini client is used when only its key is present."""
    mocker.patch.dict('os.environ', {"GOOGLE_API_KEY": "fake_google_key"})
    mock_genai = mocker.patch('google.generativeai.GenerativeModel')
    mocker.patch('google.generativeai.configure')
    client = OcrClient()
    assert client.client_type == "google"
    assert client.model == 'gemini-1.5-flash'

def test_client_no_keys_raises_error(mocker):
    """Tests that a ConfigurationError is raised if no API keys are found."""
    mocker.patch.dict('os.environ', {}, clear=True)
    with pytest.raises(ConfigurationError, match="Neither OPENROUTER_API_KEY nor GOOGLE_API_KEY"):
        OcrClient()

def test_openrouter_ocr_call(mocker):
    """Tests a successful OCR call using the mocked OpenRouter client."""
    mocker.patch.dict('os.environ', {"OPENROUTER_API_KEY": "fake_or_key"})
    
    # Mock the OpenAI client instance and its chain of calls
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "OCR text from OpenRouter"
    
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = mock_completion
    mocker.patch('openai.OpenAI', return_value=mock_openai_client)
    
    # We need a dummy file for the function to open
    mocker.patch("builtins.open", mocker.mock_open(read_data=b'fakeimagedata'))

    client = OcrClient()
    result = client.process_image_for_ocr("dummy_path.png")

    assert result == "OCR text from OpenRouter"
    mock_openai_client.chat.completions.create.assert_called_once()

def test_google_ocr_call(mocker):
    """Tests a successful OCR call using the mocked Google Gemini client."""
    mocker.patch.dict('os.environ', {"GOOGLE_API_KEY": "fake_google_key"})
    
    mock_response = MagicMock()
    mock_response.text = "OCR text from Google"
    
    mock_gemini_model = MagicMock()
    mock_gemini_model.generate_content.return_value = mock_response
    mocker.patch('google.generativeai.GenerativeModel', return_value=mock_gemini_model)
    mocker.patch('google.generativeai.configure')

    # Mock PIL.Image.open
    mocker.patch('PIL.Image.open')

    client = OcrClient()
    result = client.process_image_for_ocr("dummy_path.png")

    assert result == "OCR text from Google"
    mock_gemini_model.generate_content.assert_called_once()

def test_ocr_api_error(mocker):
    """Tests that APIError is raised when the API call fails."""
    mocker.patch.dict('os.environ', {"GOOGLE_API_KEY": "fake_google_key"})
    
    mock_gemini_model = MagicMock()
    mock_gemini_model.generate_content.side_effect = Exception("API limit reached")
    mocker.patch('google.generativeai.GenerativeModel', return_value=mock_gemini_model)
    mocker.patch('google.generativeai.configure')
    mocker.patch('PIL.Image.open')

    client = OcrClient()
    with pytest.raises(APIError, match="API limit reached"):
        client.process_image_for_ocr("dummy_path.png") 