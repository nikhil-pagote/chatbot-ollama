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
https://www.glukhov.org/post/2024/12/ollama-cheatsheet/