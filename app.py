#!/usr/bin/env python3
import logging
import gradio as gr
from ollama import Client
from ollama_mcpo_adapter import OllamaMCPOAdapter

# ── Optional: Suppress adapter logs if noisy ───────────────────────────────
logging.getLogger("ollama_mcpo_adapter").setLevel(logging.WARNING)

# ---- Configuration ----
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama2:13b"
NEWS_MCP = {"host": "localhost", "port": 8023}
WEATHER_MCP = {"host": "localhost", "port": 8024}

# ---- Adapter setup ----
adapter_news = OllamaMCPOAdapter(host=NEWS_MCP['host'], port=NEWS_MCP['port'])
adapter_weather = OllamaMCPOAdapter(host=WEATHER_MCP['host'], port=WEATHER_MCP['port'])
# Discover tools exposed by each MCP server
tools_news = adapter_news.list_tools_ollama()
tools_weather = adapter_weather.list_tools_ollama()
ALL_TOOLS = tools_news + tools_weather

# ---- Ollama client ----
client = Client(host=OLLAMA_HOST)

# ---- Core chat function (synchronous) ----
def chat_with_tools(user_message: str) -> str:
    """
    Send `user_message` to the Ollama model with MCP tools,
    dispatch any tool calls, and return the final text.
    """
    # Call the model (synchronous)
    response = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": user_message}],
        tools=ALL_TOOLS
    )

    # If Ollama requested any tools, dispatch them
    if getattr(response.message, 'tool_calls', None):
        results = []
        # dispatch calls to each adapter
        results.extend(adapter_news.call_tools_from_response(response.message.tool_calls))
        results.extend(adapter_weather.call_tools_from_response(response.message.tool_calls))
        # merge tool outputs with assistant content if any
        assistant_content = response.message.content or ""
        # Combine assistant text + tool data
        return assistant_content + "\n" + "\n".join(results)

    # Otherwise return plain assistant content
    return response.message.content or ""

# Synchronous wrapper for Gradio
# (no asyncio needed now)
def query_model(message: str) -> str:
    return chat_with_tools(message)

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("## 🗞️ News & ☁️ Weather Chatbot")
    chatbot = gr.Chatbot(type="messages")
    txt = gr.Textbox(
        placeholder="Type here and press Enter",
        show_label=False,
        container=False,
        scale=1
    )

    def respond(message, history):
        # Use messages format: list of dicts
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "…"})
        reply = query_model(message)
        history[-1]["content"] = reply
        return history, ""  # clear input

    txt.submit(respond, [txt, chatbot], [chatbot, txt])
    gr.Button("Clear").click(lambda: None, None, chatbot)

if __name__ == "__main__":
    demo.launch()
