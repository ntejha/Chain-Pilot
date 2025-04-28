#!/usr/bin/env python3
import logging
import gradio as gr
from ollama import Client
from ollama_mcpo_adapter import OllamaMCPOAdapter

logging.getLogger("ollama_mcpo_adapter").setLevel(logging.WARNING)

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama2:13b"
NEWS_MCP = {"host": "localhost", "port": 8023}
WEATHER_MCP = {"host": "localhost", "port": 8024}

adapter_news    = OllamaMCPOAdapter(**NEWS_MCP)
adapter_weather = OllamaMCPOAdapter(**WEATHER_MCP)
ALL_TOOLS = adapter_news.list_tools_ollama() + adapter_weather.list_tools_ollama()
client = Client(host=OLLAMA_HOST)

def chat_with_tools(user_message: str) -> str:
    # 1) System prompt to set expert persona
    system_prompt = {
        "role": "system",
        "content": (
            "You are “Logistics Guru,” a senior supply-chain risk manager with 15+ years of experience. "
            "When answering, cite relevant frameworks (e.g., ISO 31000), assess supplier risk factors—"
            "financial, geopolitical, ESG—and provide clear mitigation recommendations."
        )
    }

    # 2) Send messages with tools enabled
    response = client.chat(
        model=MODEL_NAME,
        messages=[system_prompt, {"role": "user", "content": user_message}],
        tools=ALL_TOOLS
    )

    # 3) Dispatch any MCP tool calls
    if getattr(response.message, 'tool_calls', None):
        results = []
        results += adapter_news.call_tools_from_response(response.message.tool_calls)
        results += adapter_weather.call_tools_from_response(response.message.tool_calls)
        assistant_text = response.message.content or ""
        return assistant_text + "\n\n" + "\n".join(results)

    return response.message.content or ""

def query_model(message: str) -> str:
    return chat_with_tools(message)

with gr.Blocks() as demo:
    gr.Markdown("## 🗞️ News & ☁️ Weather Chatbot  \n*Now speaking as a Supply-Chain Risk Expert*")
    chatbot = gr.Chatbot(type="messages")
    txt = gr.Textbox(placeholder="Ask about supplier risk, logistics, or market data…", show_label=False)

    def respond(message, history):
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "…"})
        reply = query_model(message)
        history[-1]["content"] = reply
        return history, ""

    txt.submit(respond, [txt, chatbot], [chatbot, txt])
    gr.Button("Clear").click(lambda: None, None, chatbot)

if __name__ == "__main__":
    demo.launch()
