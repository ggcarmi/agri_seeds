import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pdf_processor import PDFProcessor
from anomaly_detector import AnomalyDetector, IsolationForestStrategy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import json

class AgriAnalytics:
    def __init__(self):
        self.futures_data = None
        self.pdf_processor = PDFProcessor()
        self.anomaly_detector = AnomalyDetector(IsolationForestStrategy())
        self.pdf_text = None
        self.pdf_tables = None
        self.cv_tables = None
        from report_summarizer import ReportSummarizer
        self.summarizer = ReportSummarizer()

    def load_futures_data(self, file_path):
        """Load and preprocess futures price data"""
        self.futures_data = pd.read_csv(file_path, sep=',')
        self.futures_data['Date'] = pd.to_datetime(self.futures_data['Date'], format='%d/%m/%Y')
        return self.futures_data

    def extract_pdf_content(self, pdf_path):
        """Extract text and tables from PDF using multiple extraction strategies"""
        self.pdf_text, self.pdf_tables, self.cv_tables = self.pdf_processor.process_pdf(pdf_path)
        return self.pdf_text, self.pdf_tables, self.cv_tables

    def detect_anomalies(self, column='Close', **kwargs):
        """Detect anomalies in futures prices using configured strategy"""
        if self.futures_data is None:
            raise ValueError("Futures data not loaded")

        # Prepare data for anomaly detection
        data = self.futures_data[column]
        
        # Detect anomalies using the strategy
        result = self.anomaly_detector.detect(data, **kwargs)
        
        # Add results to the dataframe
        self.futures_data['is_anomaly'] = result['labels']
        self.futures_data['anomaly_score'] = result['scores']
        
        return self.futures_data[self.futures_data['is_anomaly'] == -1], result['stats']

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
        
        plt.title('Futures Prices with Anomalies')
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

    def generate_report_summary(self, max_length=1000):
        """Generate a structured summary of the PDF report using Llama-2 model"""
        if not self.pdf_text:
            raise ValueError("No PDF text available for summarization")

        summary = self.summarizer.summarize(self.pdf_text)
        summary_json = json.loads(self.summarizer.to_json(summary))

        summary_json.update({
            "extracted_tables_count": len(self.pdf_tables),
            "cv_detected_tables_count": len(self.cv_tables) if self.cv_tables else 0,
            "key_metrics": self._extract_key_metrics()
        })

        return json.dumps(summary_json, indent=2)

    def _extract_key_metrics(self):
        """Extract key metrics from the PDF text and tables"""
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