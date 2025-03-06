from abc import ABC, abstractmethod
from pypdf import PdfReader
import tabula
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from typing import List, Dict, Any, Tuple

class ExtractionStrategy(ABC):
    """Abstract base class for PDF extraction strategies"""
    @abstractmethod
    def extract(self, pdf_path: str) -> Any:
        pass

class TextExtractionStrategy(ExtractionStrategy):
    """Strategy for extracting text from PDF"""
    def extract(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

class TableExtractionStrategy(ExtractionStrategy):
    """Strategy for extracting tables using tabula-py"""
    def extract(self, pdf_path: str) -> List[Any]:
        return tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)

class OCRExtractionStrategy(ExtractionStrategy):
    """Strategy for extracting text using OCR"""
    def extract(self, pdf_path: str) -> List[Dict[str, Any]]:
        images = convert_from_path(pdf_path)
        cv_tables = []
        
        for image in images:
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 100:
                    table_region = opencv_image[y:y+h, x:x+w]
                    table_text = pytesseract.image_to_data(table_region, output_type=pytesseract.Output.DICT)
                    cv_tables.append(table_text)
        
        return cv_tables

class PDFProcessor:
    """Main class for processing PDF documents using different extraction strategies"""
    def __init__(self):
        self.text_strategy = TextExtractionStrategy()
        self.table_strategy = TableExtractionStrategy()
        self.ocr_strategy = OCRExtractionStrategy()
        self.extracted_text = None
        self.extracted_tables = None
        self.ocr_tables = None

    def process_pdf(self, pdf_path: str) -> Tuple[str, List[Any], List[Dict[str, Any]]]:
        """Process PDF using all available extraction strategies"""
        self.extracted_text = self.text_strategy.extract(pdf_path)
        self.extracted_tables = self.table_strategy.extract(pdf_path)
        self.ocr_tables = self.ocr_strategy.extract(pdf_path)
        
        return self.extracted_text, self.extracted_tables, self.ocr_tables

    def get_extracted_text(self) -> str:
        """Get the extracted text content"""
        return self.extracted_text if self.extracted_text else ''

    def get_extracted_tables(self) -> List[Any]:
        """Get the extracted tables"""
        return self.extracted_tables if self.extracted_tables else []

    def get_ocr_tables(self) -> List[Dict[str, Any]]:
        """Get the OCR-extracted tables"""
        return self.ocr_tables if self.ocr_tables else []