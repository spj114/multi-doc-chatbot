# Multi-Doc Chatbot 📝🤖

This chatbot allows users to interact with multiple document formats using Google Gemini AI.  
It can process **PDFs, Word, Excel, PowerPoint, and images**, extracting and storing text for intelligent searching.

## ✨ Features
- 📄 **Supports multiple document formats** (PDF, DOCX, XLSX, PPTX, Images)
- 🎨 **Extracts text from images** using Gemini Vision
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
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 Setting Up Google API Key

To use Google Gemini AI, you need to generate an API key:

1. Go to [Google AI Studio](https://aistudio.google.com/) and sign in.
2. Navigate to **API Keys** in the dashboard.
3. Click **Create API Key** and copy the generated key.
4. **Create a `.env` file** in the project directory and add the following line:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## 🎯 Usage

Run the chatbot using Streamlit:
```bash
streamlit run chatdoc.py
```

For the GUI version, run:
```bash
python chatdoc_desktop.py
```

Now you can upload documents and start chatting with them! 🌟


