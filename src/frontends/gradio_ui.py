# /src/frontends/gradio_ui.py
import gradio as gr
import requests

def chat_with_ollama(message, history):
    try:
        response = requests.post(
            "http://localhost:8000/ask",  # <-- This is your FastAPI endpoint
            json={"message": message},
            timeout=10
        )
        response.raise_for_status()
        reply = response.json().get("reply", "[No reply]")
        history.append((message, reply))
        return "", history
    except Exception as e:
        history.append((message, f"[Error: {e}]"))
        return "", history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask something...")
    msg.submit(chat_with_ollama, [msg, chatbot], [msg, chatbot])

demo.launch(share=True)
