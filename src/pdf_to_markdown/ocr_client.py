import os
from PIL import Image
import base64
import io
from .logger import logger
from .exceptions import ConfigurationError, APIError

class OcrClient:
    """
    A client to interact with an OCR API, supporting either Google Gemini
    or an OpenRouter-compatible endpoint.
    """

    def __init__(self, model: str = None):
        """
        Initializes the OCR client.
        It prioritizes OpenRouter if 'OPENROUTER_API_KEY' is set,
        otherwise falls back to Google Gemini with 'GOOGLE_API_KEY'.

        Args:
            model (str, optional): The model to use for OCR. 
                                   Defaults to 'gemini-1.5-flash'.
        """
        self.client_type = None
        self.client = None
        self.model = model

        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")

        if openrouter_key:
            self.client_type = "openrouter"
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
            self.model = model or "google/gemini-2.5-flash"
            logger.info(f"Using OpenRouter client with model: {self.model}")
        elif google_key:
            self.client_type = "google"
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            self.model = model or 'gemini-2.5-flash'
            self.client = genai.GenerativeModel(self.model)
            logger.info(f"Using Google Gemini client with model: {self.model}")
        else:
            msg = "Neither OPENROUTER_API_KEY nor GOOGLE_API_KEY environment variables are set."
            logger.error(msg)
            raise ConfigurationError(msg)

    def _prepare_image(self, image_path: str) -> str:
        """Encodes a local image into a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_image_for_ocr(self, image_path: str) -> str:
        """
        Uploads an image and asks the configured model to perform OCR.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The OCR text result from the model.
        """
        prompt = (
            "Extract all text from this image and preserve the document structure, "
            "including headings, paragraphs, lists, and tables. Format the output "
            "as structured text that can be easily converted to Markdown."
        )

        try:
            if self.client_type == "google":
                img = Image.open(image_path)
                response = self.client.generate_content([prompt, img])
                return response.text if response and response.text else ""

            elif self.client_type == "openrouter":
                base64_image = self._prepare_image(image_path)
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )
                return response.choices[0].message.content if response.choices else ""

        except FileNotFoundError as e:
            msg = f"The file was not found at {image_path}"
            logger.error(msg)
            raise APIError(msg) from e
        except Exception as e:
            msg = f"An error occurred while processing the image with the API: {e}"
            logger.error(msg)
            raise APIError(msg) from e
        
        return "" 