# PDF to Markdown Converter

English | [中文](README_CN.md)

This is a Python-based command-line tool for converting PDF documents into well-formatted Markdown files. It uses the PyMuPDF (`fitz`) library to extract PDF pages as images, then performs OCR (Optical Character Recognition) on the images via Google Gemini or OpenRouter API, and finally consolidates the recognized structured text into a single Markdown file.

## ✨ Features

- **High-precision OCR**: Utilizes the powerful Gemini 2.5 Flash model for text recognition.
- **Multiple API Support**: Configurable to use either Google Gemini or OpenRouter API through environment variables.
- **Robust Error Handling**: Includes retry mechanisms and clear error logging.
- **Easy to Use**: Simple command-line interface with clear options.
- **Automation-friendly**: Easy integration into scripts or automated workflows.
- **Code Quality**: Over 90% unit test coverage.

## ⚙️ Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Clone the repository** (if obtained via git):
    ```bash
    git clone https://github.com/JokerQianwei/PDF2Markwon-Gemini.git
    cd ./PDF2Markwon-Gemini
    ```

2.  **Install dependencies**:
    Make sure you have Poetry installed, then run:
    ```bash
    poetry install
    ```

## 🔑 API Key Configuration

Before using this tool, you need to configure API keys. The tool searches for environment variables in the following priority order:

1.  `OPENROUTER_API_KEY`: If this variable is set, OpenRouter API will be used.
2.  `GOOGLE_API_KEY`: If OpenRouter key is not set, Google Gemini API will be used.

You can set these variables in one of two ways:

### 1. (Recommended) Using `.env` File

You can create a file named `.env` in the project's root directory and define your API keys in it. This is the recommended approach as it doesn't pollute your shell's global environment.

```
# .env file content
OPENROUTER_API_KEY="sk-or-v1-..."
# or
# GOOGLE_API_KEY="..."
```

The tool will automatically load this file on startup.

### 2. Setting Environment Variables

Set your keys as environment variables. For example, add the following lines to your `.zshrc` or `.bashrc` file:

```bash
# Using OpenRouter 
export OPENROUTER_API_KEY="sk-or-v1-..."

# or using Google
export GOOGLE_API_KEY="..."
```

If neither key is set, the program will not run.

## 🚀 Usage

You can execute this tool via `poetry run`.

### Basic Usage

The simplest usage is to provide a path to an input PDF file. The program will automatically generate a `.md` file with the same name in the same directory.

```bash
poetry run pdf2md --input-path /path/to/your/document.pdf
```

### Specifying Output Path

You can use the `-o` or `--output-path` option to specify the path for the output Markdown file.

```bash
poetry run pdf2md -i /path/to/your/document.pdf -o /path/to/your/output.md
```

### Specifying OCR Model

You can use the `-m` or `--model` option to specify the model for OCR. If not provided, the default `google/gemini-2.5-flash` will be used.

```bash
# Using another model supported by OpenRouter
poetry run pdf2md -i document.pdf -m "google/gemini-2.5-pro-preview"
```

### Verbose Logging

If you encounter issues during conversion, you can enable verbose logging mode (`--verbose` or `-v`), which will print more detailed debugging information.

```bash
poetry run pdf2md -i document.pdf -v
```

### Help Information

View all available commands and options:

```bash
poetry run pdf2md --help
```

## Examples

The `PDF` folder in the project contains sample PDF files that you can use directly to test the conversion functionality. For example, to convert `HNeRV.pdf`:

```bash
poetry run pdf2md -i PDF/HNeRV.pdf
```

The converted Markdown file will be saved as `PDF/HNeRV.md` by default.


