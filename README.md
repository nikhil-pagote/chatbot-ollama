```bash
sudo apt update
sudo apt install curl -y
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry new chatbot-ollama
cd chatbot-ollama
poetry add fastapi uvicorn requests python-dotenv jinja2 aiofiles
poetry add --group dev black isort
mkdir -p src/chatbot_ollama/static
mkdir -p src/chatbot_ollama/templates
touch src/chatbot_ollama/main.py
docker exec -it ollama bash
ollama pull llama3
```
https://www.glukhov.org/post/2024/12/ollama-cheatsheet/\

On updating dependencies in pyproject.toml:
poetry lock
poetry install

To Run your FastAPI app:
poetry run uvicorn src.chatbot_ollama.main:app --reload --host 0.0.0.0
FastAPI + popup.js at localhost:8000

To run Streamlit:
poetry run streamlit run frontends/streamlit_popup.py
Streamlit UI at localhost:8501

To Run the Reflex App:
in project root: poetry run reflex init
        A blank Reflex app
poetry run reflex run
poetry run reflex run --env prod
Clear old build and force recompile
rm -rf .web
poetry run reflex run --env prod
or
poetry run reflex run --env prod --force-compile
This will launch the app at http://localhost:3000.

To run niceGUI app:
poetry run python src/frontends/nicegui_ui.py

To run Gradio app:
poetry run python src/frontends/gradio_ui.py
