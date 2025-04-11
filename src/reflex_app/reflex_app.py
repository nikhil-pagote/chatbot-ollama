import reflex as rx
import httpx

from rxconfig import config


class ChatState(rx.State):
    """Holds the chat messages and user input."""
    messages: list[str] = []
    user_input: str = ""

    async def send(self):
        if self.user_input.strip():
            user_msg = self.user_input
            self.messages.append(f"You: {user_msg}")
            self.user_input = ""

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://localhost:8000/ask",
                        json={"message": user_msg},
                        timeout=10.0
                    )
                    response.raise_for_status()
                    data = response.json()
                    bot_reply = data.get("reply", "[No response]")
            except Exception as e:
                bot_reply = f"[Error contacting bot: {str(e)}]"

            self.messages.append(f"Bot: {bot_reply}")


def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("Chat with Ollama", size="7"),
            rx.box(
                rx.foreach(ChatState.messages, lambda m: rx.text(m, size="4")),
                height="400px",
                overflow="auto",
                border="1px solid gray",
                border_radius="lg",
                padding="2",
                margin_y="4",
            ),
            rx.hstack(
                rx.input(
                    placeholder="Ask me something...",
                    value=ChatState.user_input,
                    on_change=ChatState.set_user_input,
                    width="80%",
                ),
                rx.button("Send", on_click=ChatState.send),
                spacing="2",
            ),
        ),
        padding="4",
        max_width="600px",
        margin_x="auto",
        margin_top="8",
    )


app = rx.App()
app.add_page(index)
