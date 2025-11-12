# Installation Guide - macOS / Linux

This guide covers installation and setup for macOS and Linux systems.

## Prerequisites

- Python 3.13 or higher
- pip (Python package installer)
- API keys for the providers you want to use:
  - OpenAI API key (for OpenAI models)
  - Mistral API key (for Mistral models)

## Installation Steps

1. **Clone the repository:**

   ```bash
   git clone git@github.com:eric-montagnon/llm-chatbot.git
   cd llm-chatbot
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**

   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**

   ```bash
   # Create a .env file
   touch .env

   # Add your API keys (use your preferred text editor)
   echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
   echo "MISTRAL_API_KEY=your_mistral_api_key_here" >> .env
   ```

## Running the Application

1. **Make sure your virtual environment is activated:**

   ```bash
   source .venv/bin/activate
   ```

2. **Run the Streamlit app:**

   ```bash
   streamlit run src/main.py
   ```

3. **Access the application:**
   - The application will automatically open in your default browser
   - If not, navigate to: `http://localhost:8501`

## Type Checking

### Using Pyright (Recommended)

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run type checking
pyright
```

### Using npx (Alternative)

If you have Node.js installed, you can run Pyright without installing it globally:

```bash
npx pyright
```

## Development Mode

For development with auto-reload on file changes:

```bash
streamlit run src/app.py --server.runOnSave true
```

## Troubleshooting

### Common Issues

1. **"Command not found: streamlit"**

   - Make sure your virtual environment is activated
   - Verify installation: `pip list | grep streamlit`

2. **"API key not found" errors**

   - Check that your `.env` file exists in the project root
   - Verify the API key variable names match exactly
   - Ensure `.env` is not in `.gitignore` for local development (but never commit it!)

3. **Type checking errors**

   - Make sure you're running Pyright from the project root
   - Verify Python version: `python --version` (should be 3.13+)
   - Check virtual environment is activated

4. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Verify you're running from the project root directory

### Platform-Specific Issues

**macOS:**

- If you get SSL certificate errors, you may need to install certificates:
  ```bash
  /Applications/Python\ 3.x/Install\ Certificates.command
  ```

**Linux:**

- Make sure Python 3.13+ is installed:
  ```bash
  python3 --version
  ```
- On some systems, you may need to install `python3-venv`:
  ```bash
  sudo apt-get install python3-venv  # Debian/Ubuntu
  sudo yum install python3-venv      # RedHat/CentOS
  ```

---

[← Back to main README](README.md)
