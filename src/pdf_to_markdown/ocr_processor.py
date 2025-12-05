import time
from tqdm import tqdm
from .ocr_client import OcrClient
from .logger import logger
from .exceptions import OCRProcessingError

class OCRProcessor:
    """Processes a list of images using an OCR client."""

    def __init__(self, ocr_client: OcrClient):
        """
        Initializes the OCRProcessor.

        Args:
            ocr_client (OcrClient): An instance of the Ocr client.
        """
        self.ocr_client = ocr_client

    def process_images(
        self,
        image_paths: list[str],
        max_retries: int = 3,
        initial_delay: int = 1,
    ) -> dict[int, str]:
        """
        Processes a list of image files to extract text using the Gemini API.

        Includes retry logic with exponential backoff for failed requests.

        Args:
            image_paths (list[str]): A list of paths to the image files.
            max_retries (int): The maximum number of retries for a failed API call.
            initial_delay (int): The initial delay in seconds for the retry mechanism.

        Returns:
            dict[int, str]: A dictionary mapping the page number (1-based) to the
                            extracted OCR text.
        """
        ocr_results = {}
        
        for i, image_path in enumerate(tqdm(image_paths, desc="Processing images with OCR")):
            page_number = i + 1
            retries = 0
            delay = initial_delay
            
            while retries < max_retries:
                try:
                    result_text = self.ocr_client.process_image_for_ocr(image_path)
                    
                    if result_text:
                        ocr_results[page_number] = result_text
                        break
                    else:
                        logger.warning(f"Received empty result for {image_path}. Retrying...")
                        retries += 1
                        time.sleep(delay)
                        delay *= 2
                        
                except Exception as e:
                    logger.warning(f"Error processing {image_path}: {e}. Retrying...")
                    retries += 1
                    time.sleep(delay)
                    delay *= 2

            if page_number not in ocr_results:
                msg = f"Failed to process {image_path} after {max_retries} retries."
                logger.error(msg)
                # We can choose to either raise an exception or just log an error and continue
                # For now, we will log and continue, assigning an empty string.
                ocr_results[page_number] = ""

        return ocr_results 