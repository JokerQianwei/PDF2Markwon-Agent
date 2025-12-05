import fitz  # PyMuPDF
import os
from tqdm import tqdm
from .logger import logger
from .exceptions import PDFConversionError

class PDFConverter:
    """A class to handle the conversion of PDF documents to images."""

    def convert_to_images(self, pdf_path: str, output_dir: str, dpi: int = 300):
        """
        Converts each page of a PDF document to a high-quality PNG image.

        Args:
            pdf_path (str): The file path of the PDF document.
            output_dir (str): The directory where the output images will be saved.
            dpi (int): The resolution in dots per inch for the output images.

        Returns:
            list[str]: A list of paths to the generated image files.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_paths = []
        try:
            doc = fitz.open(pdf_path)
            
            num_pages = len(doc)
            # Use tqdm for a progress bar
            for page_num in tqdm(range(num_pages), desc="Converting PDF to images"):
                page = doc.load_page(page_num)
                
                # Set the transformation matrix for the desired DPI
                zoom = dpi / 72  # 72 is the default DPI
                matrix = fitz.Matrix(zoom, zoom)
                
                pix = page.get_pixmap(matrix=matrix)
                
                image_filename = f"page_{page_num + 1:04d}.png"
                image_path = os.path.join(output_dir, image_filename)
                
                pix.save(image_path)
                image_paths.append(image_path)
                
            doc.close()
            logger.info(f"Successfully converted {num_pages} pages to images in '{output_dir}'.")

        except Exception as e:
            # Catching a broad exception here because PyMuPDF can raise
            # various errors (FileNotFoundError, ValueError for closed docs, etc.)
            msg = f"Error processing PDF '{pdf_path}': {e}"
            logger.error(msg)
            raise PDFConversionError(msg) from e

        return image_paths 