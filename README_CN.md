# PDF to Markdown 转换器

[English](README.md) | 中文

这是一个基于 Python 的命令行工具，用于将 PDF 文档转换为格式良好的 Markdown 文件。它使用 PyMuPDF (`fitz`) 库将 PDF 页面提取为图像，然后通过 Google Gemini 或 OpenRouter API 对图像进行 OCR（光学字符识别），最后将识别出的结构化文本整合成一个单一的 Markdown 文件。

## ✨ 功能特性

- **高精度 OCR**: 利用强大的 Gemini 2.5 Flash 模型进行文本识别。
- **支持多种 API**: 可通过环境变量配置使用 Google Gemini 或 OpenRouter API。
- **健壮的错误处理**: 包含重试机制和清晰的错误日志。
- **易于使用**: 简洁的命令行界面，提供明确的选项。
- **自动化**: 能够轻松集成到脚本或自动化工作流中。
- **代码质量**: 拥有超过 90% 的单元测试覆盖率。

## ⚙️ 安装

本项目使用 [Poetry](https://python-poetry.org/) 进行依赖管理。

1.  **克隆仓库** (如果您是通过 git 获取的):
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **安装依赖**:
    请确保您已经安装了 Poetry，然后运行：
    ```bash
    poetry install
    ```

## 🔑 API 密钥配置

在使用本工具前，您需要配置 API 密钥。工具会按照以下优先级顺序查找环境变量：

1.  `OPENROUTER_API_KEY`: 如果设置了此变量，将使用 OpenRouter API。
2.  `GOOGLE_API_KEY`: 如果未设置 OpenRouter 密钥，将使用 Google Gemini API。

您可以通过以下两种方式之一来设置这些变量：

### 1. (推荐) 使用 `.env` 文件

您可以在项目的根目录下创建一个名为 `.env` 的文件，并在其中定义您的 API 密钥。这是推荐的方式，因为它不会污染您 shell 的全局环境。

```
# .env 文件内容
OPENROUTER_API_KEY="sk-or-v1-..."
# 或者
# GOOGLE_API_KEY="..."
```

工具在启动时会自动加载此文件。

### 2. 设置环境变量

请将您的密钥设置为环境变量。例如，在您的 `.zshrc` 或 `.bashrc` 文件中添加如下行：

```bash
# 使用 OpenRouter 
export OPENROUTER_API_KEY="sk-or-v1-..."

# 或者使用 Google
export GOOGLE_API_KEY="..."
```

如果两个密钥都未设置，程序将无法运行。

## 🚀 使用方法

您可以通过 `poetry run` 来执行此工具。

### 基本用法

最简单的用法是提供一个输入 PDF 文件的路径。程序将自动在同一目录下生成一个同名的 `.md` 文件。

```bash
poetry run pdf2md --input-path /path/to/your/document.pdf
```

### 指定输出路径

您可以使用 `-o` 或 `--output-path` 选项来指定输出 Markdown 文件的路径。

```bash
poetry run pdf2md -i /path/to/your/document.pdf -o /path/to/your/output.md
```

### 指定 OCR 模型

您可以使用 `-m` 或 `--model` 选项来指定进行 OCR 的模型。如果未提供，将使用默认的 `google/gemini-2.5-flash-preview-05-20`。

```bash
# 使用 OpenRouter 支持的另一个模型
poetry run pdf2md -i document.pdf -m "google/gemini-2.5-pro-preview"
```

### 详细日志

如果您在转换过程中遇到问题，可以开启详细日志模式 (`--verbose` 或 `-v`)，它会打印出更详细的调试信息。

```bash
poetry run pdf2md -i document.pdf -v
```

### 帮助信息

查看所有可用的命令和选项：

```bash
poetry run pdf2md --help
```

## 示例

项目中的 `PDF` 文件夹包含了一些示例 PDF 文件，您可以直接使用它们来测试转换功能。例如，要转换 `HNeRV.pdf`：

```bash
poetry run pdf2md -i PDF/HNeRV.pdf
```

转换后的 Markdown 文件将默认保存在 `PDF/HNeRV.md`。

---
*此项目由 AI 编程助手驱动开发*

