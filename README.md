# AgriSeeds Analytics

A comprehensive agricultural analytics system for processing and analyzing agricultural market data, focusing on oilseeds futures prices and market reports.

## Features

- PDF text and table extraction with multiple methods (PDF parsing, tabula-py, and OCR)
- Futures prices analysis with EDA and anomaly detection
- Report summarization using LLM (Llama-2 model)
- Interactive Jupyter notebook for demonstration

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agri_seeds.git
cd agri_seeds
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment:
   - Make sure you have Python 3.8+ installed
   - Install Tesseract OCR for PDF table extraction
   - Download and set up Llama-2 model credentials if needed

## Usage

1. Run the Jupyter notebook demo:
```bash
jupyter notebook agri_demo.ipynb
```

2. Use the Python modules directly:
```python
from agri_analytics import AgriAnalytics

analytics = AgriAnalytics()
futures_data = analytics.load_futures_data('futures_prices.csv')
analytics.detect_anomalies()
```

## Project Structure

- `agri_analytics.py`: Main analytics class and utilities
- `pdf_processor.py`: PDF extraction strategies
- `anomaly_detector.py`: Time series anomaly detection
- `report_summarizer.py`: LLM-based report summarization
- `agri_demo.ipynb`: Interactive demo notebook

## Dependencies

See `requirements.txt` for a full list of dependencies.