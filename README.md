# 💬 LLM Chatbot

A minimal yet powerful chatbot application built with Streamlit that supports multiple LLM providers (OpenAI and Mistral). The application features a clean interface, streaming responses, and persistent chat history during sessions.

## Features

- 🔄 Support for multiple LLM providers (OpenAI, Mistral)
- 💬 Real-time streaming responses
- 🎨 Clean and intuitive UI built with Streamlit
- 📝 Persistent chat history during sessions
- ⚙️ Customizable system prompts
- 🔧 Type checking with Pyright
- 🎯 Model selection for each provider

## Quick Start

Choose your platform for detailed installation instructions:

- 🍎 [**macOS / Linux Installation Guide**](INSTALL_MAC_LINUX.md)
- 🪟 [**Windows Installation Guide**](INSTALL_WINDOWS.md)

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- API keys for the providers you want to use:
  - OpenAI API key (for OpenAI models)
  - Mistral API key (for Mistral models)

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
```

### Pyright Configuration

The project includes type checking configuration in two files:

- `pyrightconfig.json` - Main Pyright configuration
- `pyproject.toml` - Python project configuration including Pyright settings

Both configurations are set to use Python 3.9+ and check the `src` directory.

## Usage

After installation, refer to your platform-specific guide for running the application:

- [Running on macOS / Linux](INSTALL_MAC_LINUX.md#running-the-application)
- [Running on Windows](INSTALL_WINDOWS.md#running-the-application)

The application will automatically open in your default browser at `http://localhost:8501`.

## Type Checking

The project uses Pyright for static type checking.

### Using npx (Cross-Platform)

If you have Node.js installed, you can run Pyright without installing it globally:

```bash
npx pyright
```

### Platform-Specific Instructions

For detailed type checking instructions for your platform:

- [Type Checking on macOS / Linux](INSTALL_MAC_LINUX.md#type-checking)
- [Type Checking on Windows](INSTALL_WINDOWS.md#type-checking)

### Type Checking Configuration

The type checking is configured with:

- **Type Checking Mode**: Standard
- **Python Version**: 3.9+
- **Included Paths**: `src/**/*.py`
- **Excluded Paths**: `__pycache__`, `.venv`, `node_modules`, etc.

View detailed configuration in `pyrightconfig.json` and `pyproject.toml`.

## Project Structure

```
llm-chatbot/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── chat_manager.py        # Chat session management
│   ├── config.py              # Configuration settings
│   ├── ui_components.py       # UI components
│   └── providers/
│       ├── __init__.py
│       ├── base.py            # Base provider interface
│       ├── openai_provider.py # OpenAI integration
│       └── mistral_provider.py # Mistral integration
├── requirements.txt           # Python dependencies
├── pyrightconfig.json        # Pyright type checking config
├── pyproject.toml            # Python project configuration
├── INSTALL_MAC_LINUX.md      # macOS/Linux installation guide
├── INSTALL_WINDOWS.md        # Windows installation guide
└── README.md                 # This file
```

## Development

### Adding a New LLM Provider

1. Create a new provider class in `src/providers/` that inherits from `BaseLLMProvider`
2. Implement the required methods: `chat()`, `chat_stream()`, `list_models()`
3. Register the provider in the configuration
4. Update the UI to include the new provider option

### Running in Development Mode

For platform-specific development instructions, see:

- [macOS / Linux Development](INSTALL_MAC_LINUX.md#development-mode)
- [Windows Development](INSTALL_WINDOWS.md#development-mode)

## Troubleshooting

For platform-specific troubleshooting, please refer to:

- [macOS / Linux Troubleshooting](INSTALL_MAC_LINUX.md#troubleshooting)
- [Windows Troubleshooting](INSTALL_WINDOWS.md#troubleshooting)

### Common Issues (All Platforms)

1. **"Command not found: streamlit"**

   - Make sure your virtual environment is activated
   - Verify installation: `pip list | grep streamlit` (macOS/Linux) or `pip list | findstr streamlit` (Windows)

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

## Dependencies

- `streamlit>=1.36` - Web application framework
- `openai>=1.30.0` - OpenAI API client
- `mistralai>=1.0.0` - Mistral AI API client
- `python-dotenv>=1.0.1` - Environment variable management
- `pyright>=1.1.350` - Static type checker

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

[Add support/contact information here]
