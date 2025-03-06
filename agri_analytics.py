import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pypdf import PdfReader
import tabula
import cv2
import numpy as np
from PIL import Image
from sklearn.ensemble import IsolationForest
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import json
import pytesseract
from pdf2image import convert_from_path
from report_summarizer import ReportSummarizer


class AgriAnalytics:
    def __init__(self):
        self.futures_data = None
        self.pdf_text = None
        self.pdf_tables = None
        self.anomaly_detector = None
        self.table_extractor = None
        self.summarizer = ReportSummarizer()

    def detect_anomalies(self, column='Close', contamination=0.1, n_estimators=100, max_samples='auto'):
        """Detect anomalies in futures prices using Isolation Forest with configurable parameters"""
        if self.futures_data is None:
            raise ValueError("Futures data not loaded")

        # Prepare data for anomaly detection
        X = self.futures_data[column].values.reshape(-1, 1)
        
        # Initialize and fit the anomaly detector with configurable parameters
        self.anomaly_detector = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42
        )
        anomalies = self.anomaly_detector.fit_predict(X)
        
        # Add anomaly labels and scores to the dataframe
        self.futures_data['is_anomaly'] = anomalies
        self.futures_data['anomaly_score'] = self.anomaly_detector.score_samples(X)
        
        # Calculate additional statistics for anomalies
        anomaly_data = self.futures_data[self.futures_data['is_anomaly'] == -1]
        stats = {
            'count': len(anomaly_data),
            'mean_price': anomaly_data[column].mean(),
            'std_price': anomaly_data[column].std(),
            'min_price': anomaly_data[column].min(),
            'max_price': anomaly_data[column].max()
        }
        
        return anomaly_data, stats

    def visualize_prices_and_anomalies(self):
        """Create enhanced visualization of prices and detected anomalies"""
        plt.figure(figsize=(15, 10))
        
        # Create subplot for price and anomalies
        plt.subplot(2, 1, 1)
        plt.plot(self.futures_data['Date'], self.futures_data['Close'], 
                 label='Close Price', color='blue', alpha=0.6)
        
        # Plot anomalies with larger markers
        anomalies = self.futures_data[self.futures_data['is_anomaly'] == -1]
        plt.scatter(anomalies['Date'], anomalies['Close'], 
                    color='red', label='Anomaly', s=100)
        
        plt.title('Soybean Futures Prices with Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create subplot for anomaly scores
        plt.subplot(2, 1, 2)
        plt.plot(self.futures_data['Date'], self.futures_data['anomaly_score'], 
                 color='purple', alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title('Anomaly Scores Over Time')
        plt.xlabel('Date')
        plt.ylabel('Anomaly Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('price_anomalies.png', dpi=300, bbox_inches='tight')
        plt.close()

    def load_futures_data(self, file_path):
        """Load and preprocess futures price data"""
        self.futures_data = pd.read_csv(file_path, sep=',')
        # Convert date to datetime
        self.futures_data['Date'] = pd.to_datetime(self.futures_data['Date'], format='%d/%m/%Y')
        return self.futures_data

    def extract_pdf_content(self, pdf_path):
        """Extract text and tables from PDF using multiple methods for better accuracy"""
        # Extract text using PdfReader
        reader = PdfReader(pdf_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        self.pdf_text = text
        
        print("\n=== Extracted Text from PDF ===\n")
        print(text)
        print("\n=== End of Text ===\n")

        # Extract tables using tabula-py
        self.pdf_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        
        print("\n=== Extracted Tables from PDF ===\n")
        for i, table in enumerate(self.pdf_tables):
            print(f"\nTable {i+1}:")
            print(table.to_string())
        print("\n=== End of Tables ===\n")

        # Convert PDF to images for OCR and CV-based table detection
        images = convert_from_path(pdf_path)
        cv_tables = []
        
        for i, image in enumerate(images):
            # Convert PIL image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect tables using image processing
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be tables
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 100:  # Minimum size threshold for tables
                    table_region = opencv_image[y:y+h, x:x+w]
                    # Extract text from table region using OCR
                    table_text = pytesseract.image_to_data(table_region, output_type=pytesseract.Output.DICT)
                    cv_tables.append(table_text)
        
        print("\n=== OCR-detected Tables ===\n")
        for i, table in enumerate(cv_tables):
            print(f"\nOCR Table {i+1}:")
            # Convert OCR output to a more readable format
            text_data = [word for word in table['text'] if word.strip() != '']
            print(' '.join(text_data))
        print("\n=== End of OCR Tables ===\n")
        
        return self.pdf_text, self.pdf_tables, cv_tables

    def generate_report_summary(self, max_length=1000):
        """Generate a structured summary of the PDF report using Llama-2 model"""
        if not self.pdf_text:
            raise ValueError("No PDF text available for summarization")

        # Generate structured summary using ReportSummarizer
        summary = self.summarizer.summarize(self.pdf_text)
        summary_json = json.loads(self.summarizer.to_json(summary))

        # Add additional metadata
        summary_json.update({
            "extracted_tables_count": len(self.pdf_tables),
            "cv_detected_tables_count": len(self.cv_tables) if hasattr(self, 'cv_tables') else 0,
            "key_metrics": self._extract_key_metrics()
        })

        return json.dumps(summary_json, indent=2)

    def _extract_key_metrics(self):
        """Extract key metrics from the PDF text and tables"""
        # This is a placeholder for metric extraction logic
        # In a real implementation, you would add specific metric extraction rules
        metrics = {
            "price_trends": {},
            "market_indicators": {},
            "forecast_values": {}
        }
        return metrics

def main():
    # Initialize the analytics system
    analytics = AgriAnalytics()

    # Load and process futures data
    futures_data = analytics.load_futures_data('futures_prices.csv')
    print("Futures data loaded successfully")

    # Extract PDF content with enhanced methods
    pdf_text, pdf_tables, cv_tables = analytics.extract_pdf_content('Oilseeds Outlook Report 122024.pdf')
    print(f"PDF content extracted successfully: {len(pdf_tables)} tables found")

    # Generate report summary
    summary_json = analytics.generate_report_summary()
    print("Report summary generated successfully")

    # Detect anomalies with custom parameters
    anomalies = analytics.detect_anomalies(contamination=0.1, n_estimators=150)
    print(f"Found {len(anomalies)} anomalies in the price data")

    # Visualize results
    analytics.visualize_prices_and_anomalies()
    print("Visualization saved as 'price_anomalies.png'")

    # Save summary to file
    with open('report_summary.json', 'w') as f:
        f.write(summary_json)
    print("Summary saved as 'report_summary.json'")

if __name__ == "__main__":
    main()