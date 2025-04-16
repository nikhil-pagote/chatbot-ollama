from pathlib import Path

cheatsheet_md = """
# 🐍 Poetry Cheat Sheet

This cheat sheet summarizes all essential commands and configuration fields used in managing Python projects with [Poetry](https://python-poetry.org/).

---

## 📦 Basic Commands

| Task | Command |
|------|---------|
| Create a new Poetry project | `poetry new my_project` |
| Initialize Poetry in an existing directory | `poetry init` |
| Install dependencies | `poetry install` |
| Add a new dependency | `poetry add requests` |
| Add a dev dependency | `poetry add --group dev black` |
| Remove a dependency | `poetry remove requests` |
| Update dependencies | `poetry update` |
| Check dependency tree | `poetry show --tree` |
| Run a command inside virtual env | `poetry run python script.py` |
| Open a shell within virtual env | `poetry shell` |
| Export to requirements.txt | `poetry export -f requirements.txt --output requirements.txt` |
| Publish package | `poetry publish --build` |

---

## ⚙️ pyproject.toml - Full Example

```toml
# ============================
# Project Metadata
# ============================
[tool.poetry]
name = "my-awesome-project"                   # Name of your package
version = "0.1.0"                             # Version using semantic versioning
description = "RAG chatbot demo with LangGraph, LangChain, and Streamlit."  # Short description
authors = ["Your Name <you@example.com>"]     # List of authors
license = "MIT"                                # Open-source license
readme = "README.md"                           # Path to README used on PyPI
keywords = ["langchain", "langgraph", "rag", "chatbot"]  # For package discovery

# Optional project URLs
homepage = "https://github.com/yourusername/my-awesome-project"
repository = "https://github.com/yourusername/my-awesome-project"
documentation = "https://yourusername.github.io/my-awesome-project"

# Define package source directory
packages = [{ include = "my_awesome_project" }]

classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent"
]

# ============================
# Runtime Dependencies
# ============================
[tool.poetry.dependencies]
python = ">=3.9,<4.0"
fastapi = "^0.110.0"
uvicorn = { extras = ["standard"], version = "^0.29.0" }
langchain = "^0.1.13"
langgraph = "^0.0.35"
streamlit = "^1.32.0"
pydantic = "^2.6"
httpx = "^0.27.0"
python-dotenv = "^1.0.1"
openai = { version = "^1.14.2", optional = true }

# ============================
# Optional Features
# ============================
[tool.poetry.extras]
openai_support = ["openai"]  # Enable with `--extras openai_support`

# ============================
# Development Dependencies
# ============================
[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
black = "^24.3"
ruff = "^0.3"
mypy = "^1.9"
ipykernel = "^6.29"
jupyterlab = "^4.1"
poethepoet = "^0.24"

# ============================
# CLI Scripts
# ============================
[tool.poetry.scripts]
app = "my_awesome_project.main:run"        # Use `poetry run app` to call main.run()
cli = "my_awesome_project.cli:main"        # Use `poetry run cli` to call cli.main()

# ============================
# Task Runner: Poe the Poet
# ============================
[tool.poe.tasks]
lint = "ruff check ."
format = "black ."
typecheck = "mypy ."
test = "pytest"
dev = "streamlit run src/langgraph_chain/frontend.py"

# ============================
# Code Quality Tools
# ============================

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I"]
exclude = ["notebooks", ".venv", "__pycache__"]

[tool.mypy]
strict = true
ignore_missing_imports = true

# ============================
# LangSmith Tracing
# ============================
[tool.langchain]
project = "LangGraph_Demo"
tracing = true
endpoint = "https://api.smith.langchain.com"

# ============================
# Build Configuration
# ============================
[build-system]
requires = ["poetry-core>=1.6.1"]
build-backend = "poetry.core.masonry.api"

```

##  🧪 Poetry + VS Code
Enable Poetry in VS Code: add this to your 
```bash
.vscode/settings.json
```
```json
{
  "python.venvPath": "~/.cache/pypoetry/virtualenvs",
  "python.defaultInterpreterPath": ".venv/bin/python"
}
```
## 📁 Where Poetry Creates Virtual Envs
- By default: ~/.cache/pypoetry/virtualenvs/your-project-name-<hash>
- To use .venv inside project: poetry config virtualenvs.in-project true

## 🌱 Using uv as Poetry Backend
* Replace default installer with uv:
```bash
poetry config experimental.new-installer false
poetry config installer.parallel true
poetry config installer.uv true
```
## Notes
* requires-python ensures compatibility across environments.
* tool.poetry.scripts allows creation of CLI entrypoints.
* tool.poetry.group.dev.dependencies separates development-only dependencies.