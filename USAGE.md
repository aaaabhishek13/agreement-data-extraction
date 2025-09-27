# Quick Start Guide

## 🚀 Getting Started

### 1. Set up your API keys

Edit the `.env` file and add your API keys:

```bash
# For OpenAI GPT models
OPENAI_API_KEY=your_openai_api_key_here

# For Google Gemini models  
GEMINI_API_KEY=your_gemini_api_key_here

# Choose your default provider
DEFAULT_LLM_PROVIDER=openai  # or 'gemini'
```

### 2. Run the application

```bash
python app.py
```

The application will start at `http://localhost:5000`

### 3. Upload a lease agreement

1. Open your browser to `http://localhost:5000`
2. Select your LLM provider (OpenAI or Gemini)
3. Upload a lease agreement document (PDF, DOCX, DOC, or TXT)
4. Click "Extract Data" to process the document
5. View the extracted structured data

## 📋 Supported File Formats

- **PDF** (.pdf) - Text-based PDFs work best
- **Microsoft Word** (.docx, .doc) - Modern Word documents
- **Plain Text** (.txt) - Simple text files

## 🤖 LLM Providers

### OpenAI GPT
- High accuracy for structured data extraction
- Requires OpenAI API key
- Uses GPT-4 by default (configurable)

### Google Gemini
- Google's advanced language model  
- Requires Google AI Studio API key
- Uses Gemini-Pro model

## 📊 Extracted Data Fields

The tool extracts comprehensive lease agreement data including:

- **Agreement Details**: Document info, project name, location
- **Parties**: Landlord and tenant information  
- **Unit Details**: Floor, wing, areas, specifications
- **Lease Terms**: Dates, rent, escalation, conditions
- **Financials**: Security deposit, market value, taxes
- **Parking & CAM**: Parking slots, maintenance charges
- **Property Tax**: Tax amounts and responsibilities
- **Miscellaneous**: Comments, documents, attachments

## 🔧 Troubleshooting

### Common Issues

1. **"No API key found"**
   - Add your API key to the `.env` file
   - Restart the application

2. **"File processing failed"** 
   - Ensure file is under 16MB
   - Check file format is supported
   - Verify file is not corrupted

3. **"Import errors"**
   - Run: `pip install -r requirements.txt`
   - Ensure Python virtual environment is activated

4. **"Gemini API error: Content blocked by safety filters"**
   - This happens when Gemini's safety filters flag content as potentially harmful
   - The system automatically attempts fallback to OpenAI if configured
   - Try using OpenAI provider directly for this document
   - Ensure your lease document doesn't contain sensitive personal information
   - Consider redacting personal details like SSN, phone numbers, etc.

5. **"Both providers failed"**
   - Check your API keys are valid and have sufficient quota
   - Verify your internet connection
   - Try with a simpler document first
   - Check API status pages for service outages

### API Provider Notes

**OpenAI GPT:**
- Generally more permissive with business document content
- Better handling of complex formatting
- Consistent JSON output structure

**Google Gemini:**
- More restrictive safety filters
- May block content with personal information
- Enable automatic fallback to OpenAI for best results

### Testing the System

Run the built-in test script:

```bash
python test_system.py
```

## 💡 Tips for Best Results

1. **Use clear, well-formatted lease documents**
2. **Ensure text is readable (not scanned images)**  
3. **Documents in English work best**
4. **Include complete lease agreements for full extraction**

## 🏗️ Project Structure

```
doc-extraction/
├── src/
│   ├── models/          # Pydantic data models
│   ├── services/        # Core business logic
│   └── web/            # Flask web application
├── uploads/            # Temporary file storage
├── requirements.txt    # Python dependencies
├── app.py             # Main application
├── test_system.py     # System tests
└── .env               # Environment variables
```

## 🔐 Security Notes

- API keys are stored in `.env` file (not in git)
- Uploaded files are temporarily stored and cleaned up
- No data is permanently stored on the server
- Use HTTPS in production deployments

## 📈 Performance Tips

- Smaller documents process faster
- OpenAI generally provides more consistent results
- Consider batch processing for multiple documents
- Monitor API usage and costs

## 🆘 Support

For issues or questions:
1. Check the console output for error messages
2. Verify your API keys are correct
3. Test with the system test script
4. Ensure all dependencies are installed
