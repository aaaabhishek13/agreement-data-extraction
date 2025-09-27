# Lease Agreement Data Extraction Tool

## Overview
A robust, production-ready LLM-powered tool for extracting structured data from lease agreement documents. Supports multiple document formats (PDF, DOCX) and multiple LLM providers (OpenAI GPT-4, Google Gemini) with advanced error handling and fallback mechanisms.

## Features

### 🚀 Core Functionality
- **Multi-format Document Processing**: PDF and DOCX support
- **Dual LLM Support**: OpenAI GPT-4 and Google Gemini with automatic fallback
- **Structured Data Extraction**: Comprehensive lease agreement schema
- **Web Interface**: Clean, responsive UI for document upload and result viewing

### 🛡️ Reliability & Error Handling
- **Robust JSON Parsing**: Multiple parsing strategies with auto-fixing
- **Retry Logic**: Automatic retry with exponential backoff
- **Safety Filter Handling**: Graceful handling of LLM safety restrictions
- **Comprehensive Logging**: Detailed logging with configurable verbosity

### 🏗️ Production Ready
- **VM Deployment**: Complete deployment scripts and configuration
- **Process Management**: Supervisor for auto-restart and monitoring
- **Scalable Architecture**: Multi-worker Gunicorn setup
- **Security**: Firewall configuration and secure user setup
- JSON schema-based data validation

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key_here
   GEMINI_API_KEY=your_gemini_key_here
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```
2. Open your browser to `http://localhost:5000`
3. Upload a lease agreement document
4. View the extracted structured data

## API Endpoints

- `GET /` - Main upload interface
- `POST /upload` - Upload and process document
- `GET /result/<task_id>` - Get extraction results

## Data Schema

The tool extracts data according to a comprehensive JSON schema covering:
- Agreement details
- Party information (landlord/tenant)
- Unit details and specifications
- Lease terms and conditions
- Financial information
- Parking and CAM charges
- Property tax details
- Miscellaneous information
