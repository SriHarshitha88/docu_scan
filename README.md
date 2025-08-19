# DocuScan - Document Extractor

A web-based application for extracting specific information from uploaded documents (PDFs and images).

## Phase 1: Foundational Infrastructure & Document Upload

This initial phase focuses on building a solid, scalable foundation with document upload functionality.

## Features

- Clean, user-friendly Streamlit interface
- Multi-file upload support (PDF, PNG, JPG, JPEG)
- Field specification for targeted extraction
- Structured logging with loguru
- Configuration management with environment variables
- Modular project structure for future development

## Project Structure

```
docuscan/
├── src/                    # Core application logic
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   └── logger.py          # Logging setup
├── ui/                    # User interface
│   └── app.py            # Main Streamlit application
├── data/                 # Input data storage
├── outputs/              # Processing results and logs
├── models/               # Future ML models
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure as needed
4. Run the application:
   ```bash
   streamlit run ui/app.py
   ```

## Usage

1. Open the application in your browser
2. Upload PDF files or images using the file uploader
3. Specify the fields you want to extract (e.g., "Invoice Number, Date, Total Amount")
4. Click "Process Documents" to confirm upload

## Technology Stack

- **Frontend**: Streamlit
- **Configuration**: Pydantic + python-dotenv
- **Logging**: loguru
- **Language**: Python 3.8+

## Future Phases

- Document text extraction (OCR for images, text parsing for PDFs)
- AI-powered field identification and extraction
- Multiple export formats (CSV, JSON, Excel)
- Batch processing capabilities
- Advanced document preprocessing