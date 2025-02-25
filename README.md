# Multi-Doc Chatbot 📝🤖

This chatbot allows users to interact with multiple document formats using Google Gemini AI.  
It can process **PDFs, Word, Excel, PowerPoint, and images**, extracting and storing text for intelligent searching.

## ✨ Features
- 📄 **Supports multiple document formats** (PDF, DOCX, XLSX, PPTX, Images)
- 🖼️ **Extracts text from images** using Gemini Vision
- 🔍 **Uses FAISS vector store** for efficient document search
- 💬 **Chat with your documents** using a conversational AI model

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/multi-doc-chatbot.git
   cd multi-doc-chatbot
   ```

2. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

Run the chatbot using Streamlit:
```bash
streamlit run chatdoc.py
