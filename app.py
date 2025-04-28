#!/usr/bin/env python3
import logging
import gradio as gr
from ollama import Client
from ollama_mcpo_adapter import OllamaMCPOAdapter

logging.getLogger("ollama_mcpo_adapter").setLevel(logging.WARNING)

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME   = "llama2:13b"
NEWS_MCP     = {"host": "localhost", "port": 8023}
WEATHER_MCP  = {"host": "localhost", "port": 8024}

# Setup adapters & client
adapter_news    = OllamaMCPOAdapter(**NEWS_MCP)
adapter_weather = OllamaMCPOAdapter(**WEATHER_MCP)
ALL_TOOLS       = adapter_news.list_tools_ollama() + adapter_weather.list_tools_ollama()
client          = Client(host=OLLAMA_HOST)

# Persona prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are “Logistics Guru,” a senior supply-chain risk manager with 15+ years of experience. "
        "When answering, cite relevant frameworks (e.g., ISO 31000), assess supplier risk factors—"
        "financial, geopolitical, ESG—and provide clear mitigation recommendations."
    )
}

# Max number of past turns to keep
MAX_MEMORY = 10

def chat_with_tools_and_memory(memory, user_message):
    """
    memory: list of dict messages (role/content) from prior turns
    user_message: new user content string
    """
    # Build the message list: system + past memory + new user turn
    messages = [SYSTEM_PROMPT] + memory + [{"role": "user", "content": user_message}]
    
    # Call the LLM with available tools
    response = client.chat(model=MODEL_NAME, messages=messages, tools=ALL_TOOLS)

    # Dispatch any tool calls
    tool_output = ""
    if getattr(response.message, "tool_calls", None):
        news_res   = adapter_news.call_tools_from_response(response.message.tool_calls)
        weather_res= adapter_weather.call_tools_from_response(response.message.tool_calls)
        tool_output = "\n".join(news_res + weather_res)

    assistant_reply = (response.message.content or "").strip()
    full_reply = assistant_reply + ("\n\n" + tool_output if tool_output else "")

    # Update memory: append user & assistant, trim if necessary
    memory = memory + [{"role": "user", "content": user_message},
                       {"role": "assistant", "content": assistant_reply}]
    if len(memory) > MAX_MEMORY * 2:
        # each turn is two messages; keep only the last MAX_MEMORY turns
        memory = memory[-MAX_MEMORY*2:]

    return memory, full_reply

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🗞️ News & ☁️ Weather Chatbot  \n*Now with Supply-Chain Expertise & Memory*")
    chatbot = gr.Chatbot(type="messages")
    txt     = gr.Textbox(placeholder="Ask about supplier risk, logistics, or market data…", show_label=False)
    memory  = gr.State([])  # holds past messages

    def respond(message, chat_history, memory_state):
        # chat_history used only for UI; memory_state for LLM context
        memory_state, reply = chat_with_tools_and_memory(memory_state, message)
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": reply})
        return chat_history, "", memory_state

    txt.submit(respond, [txt, chatbot, memory], [chatbot, txt, memory])
    gr.Button("Clear").click(lambda: ([], ""), None, [chatbot, txt])

if __name__ == "__main__":
    demo.launch()
