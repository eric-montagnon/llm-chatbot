# Installation Guide - Windows

This guide covers installation and setup for Windows systems.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- API keys for the providers you want to use:
  - OpenAI API key (for OpenAI models)
  - Mistral API key (for Mistral models)

## Installation Steps

1. **Clone the repository:**

   ```powershell
   git clone <repository-url>
   cd llm-chatbot
   ```

2. **Create a virtual environment:**

   ```powershell
   python -m venv .venv
   ```

3. **Activate the virtual environment:**

   ```powershell
   .venv\Scripts\activate
   ```

4. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**

   ```powershell
   # Create a .env file
   New-Item -Path .env -ItemType File

   # Add your API keys using notepad or any text editor
   notepad .env
   ```

   Add the following content to the `.env` file:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

## Running the Application

1. **Make sure your virtual environment is activated:**

   ```powershell
   .venv\Scripts\activate
   ```

2. **Run the Streamlit app:**

   ```powershell
   streamlit run src/app.py
   ```

3. **Access the application:**
   - The application will automatically open in your default browser
   - If not, navigate to: `http://localhost:8501`

## Type Checking

### Using Pyright (Recommended)

```powershell
# Activate virtual environment first
.venv\Scripts\activate

# Run type checking
pyright
```

### Using npx (Alternative)

If you have Node.js installed, you can run Pyright without installing it globally:

```powershell
npx pyright
```

## Development Mode

For development with auto-reload on file changes:

```powershell
streamlit run src/app.py --server.runOnSave true
```

## Troubleshooting

### Common Issues

1. **"Command not found: streamlit"**

   - Make sure your virtual environment is activated
   - Verify installation: `pip list | findstr streamlit`

2. **"API key not found" errors**

   - Check that your `.env` file exists in the project root
   - Verify the API key variable names match exactly
   - Ensure `.env` is not in `.gitignore` for local development (but never commit it!)

3. **Type checking errors**

   - Make sure you're running Pyright from the project root
   - Verify Python version: `python --version` (should be 3.9+)
   - Check virtual environment is activated

4. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Verify you're running from the project root directory

### Windows-Specific Issues

- If activation script doesn't work, you may need to adjust PowerShell execution policy:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

[← Back to main README](README.md)
