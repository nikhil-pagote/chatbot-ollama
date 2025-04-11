from nicegui import ui
import httpx

messages = []

def send():
    user_input = input_box.value.strip()
    if not user_input:
        return
    messages.append(f"You: {user_input}")
    input_box.value = ""
    update_messages()

    async def fetch():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8000/ask",
                    json={"message": user_input},
                    timeout=10.0
                )
                response.raise_for_status()
                data = await response.json()  # ← Fix here
                messages.append(f"Bot: {data.get('reply', '[No reply]')}")
        except Exception as e:
            messages.append(f"Bot: [Error: {str(e)}]")
        update_messages()

    ui.run_later(fetch)

def update_messages():
    message_box.clear()
    for m in messages:
        with message_box:
            ui.label(m)

ui.label("Chat with Ollama").classes("text-h4 mt-4")
message_box = ui.column().classes("scroll max-h-96 overflow-auto")
input_box = ui.input(placeholder="Type your message here...").props("dense outlined").classes("w-3/4")
ui.button("Send", on_click=send).classes("w-1/4")

ui.run(port=8080)
