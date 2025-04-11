import reflex as rx
from rxconfig import config


class ChatState(rx.State):
    """Holds the chat messages and user input."""
    messages: list[str] = []
    user_input: str = ""

    def send(self):
        if self.user_input.strip():
            self.messages.append(f"You: {self.user_input}")
            # TODO: Replace the below line with actual call to backend/chatbot
            self.messages.append(f"Bot: You said '{self.user_input}'")
            self.user_input = ""


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
