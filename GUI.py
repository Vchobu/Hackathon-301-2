import uuid

from nicegui import app, ui

from hackathon.chatbot.agent import ACCUEIL, creer_agent

AGENT = None

async def demarrer_agent():
    global AGENT
    AGENT = await creer_agent()

app.on_startup(demarrer_agent)

ui.add_head_html(
    """
    <style>
        body {
            background: #f4f7fb;
        }

        .chat-window {
            height: min(70vh, 720px);
            overflow-y: auto;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 20px;
        }

        .message {
            max-width: 80%;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            border-radius: 14px;
            padding: 10px 14px;
        }

        .message-user {
            align-self: flex-end;
            background: #2563eb;
            color: white;
        }

        .message-assistant {
            align-self: flex-start;
            background: #eef2f7;
            color: #172033;
        }

        .typing-dots {
            display: inline-flex;
            gap: 3px;
            min-width: 30px;
        }

        .typing-dots span {
            animation: typing 1.2s infinite;
            font-weight: bold;
        }

        .typing-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% {
                opacity: 0.25;
            }
            30% {
                opacity: 1;
            }
        }
    </style>
    """,
    shared=True,
)


@ui.page("/")
def chat_page() -> None:
    with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-4"):
        ui.label("Credit assistant").classes("text-2xl font-bold")

        chat_window = ui.column().classes("chat-window w-full gap-3")
        with chat_window:
            ui.label(
                ACCUEIL

            ).classes("message message-assistant")
        with chat_window:
            typing_indicator = ui.html(
                '<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>'
            ).classes("message message-assistant")
        typing_indicator.set_visibility(False)
        scroll_anchor = ui.element("div")

        with ui.row().classes("w-full items-end gap-2"):
            message_input = ui.textarea(
                label="Your message",
                placeholder="Type your message...",
            ).classes("flex-1")
            send_button = ui.button("Send", icon="send").classes("h-14")

        is_waiting = False
        thread_id = str(uuid.uuid4())   # une conversation séparée par onglet du navigateur
        conversation_history: list[tuple[str, str]] = []

    def add_message(text: str, sender: str) -> None:
        classes = (
            "message message-user" if sender == "user" else "message message-assistant"
        )
        with chat_window:
            ui.label(text).classes(classes)
        scroll_to_bottom()

    def scroll_to_bottom() -> None:
        scroll_anchor.move(chat_window)
        scroll_anchor.run_method(
            "scrollIntoView",
            {"behavior": "smooth", "block": "end"},
        )

    async def send_message(_event=None) -> None:
        nonlocal is_waiting
        question = (message_input.value or "").strip()
        if not question or is_waiting:
            return

        is_waiting = True
        add_message(question, "user")
        message_input.value = ""
        send_button.disable()
        message_input.disable()
        typing_indicator.move(chat_window)
        typing_indicator.set_visibility(True)
        scroll_to_bottom()

        try:
            reponse = await AGENT.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                {"configurable": {"thread_id": thread_id}},
            )
            answer = reponse["messages"][-1].content   # dernier message = réponse de l'agent
            conversation_history.append((question, answer))
            add_message(answer or "The assistant did not return a text response.", "assistant")
        except Exception as error:
            error_message = f"Failed to get response: {error}"
            add_message(error_message, "assistant")
            ui.notify(error_message, type="negative")
        finally:
            typing_indicator.set_visibility(False)
            message_input.enable()
            send_button.enable()
            is_waiting = False
            message_input.run_method("focus")

    send_button.on_click(send_message)
    message_input.on("keydown.enter", send_message)


if __name__ == "__main__":
    ui.run(title="Chatbot", reload=False)