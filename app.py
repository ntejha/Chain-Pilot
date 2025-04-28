import os
import gradio as gr

from ollama import Client
from ollama_mcpo_adapter import OllamaMCPOAdapter

# ─── Bootstrap RAG artifacts on first run ──────────────────────────────────────
from rag.chunker  import run as do_chunk
from rag.embedder import run as do_embed
from rag.indexer  import run as do_index

do_chunk()
do_embed()
do_index()

# ─── Retriever setup ───────────────────────────────────────────────────────────
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

EMB_MODEL = "all-MiniLM-L6-v2"
model     = SentenceTransformer(EMB_MODEL)

STORE_DIR     = os.path.join(os.path.dirname(__file__), ".chromastore")
client_chroma = PersistentClient(path=STORE_DIR)
collection    = client_chroma.get_collection(name="supply_chain")

def retrieve_context(query: str, k: int = 3):
    query_emb = model.encode(query).tolist()
    results   = collection.query(query_embeddings=[query_emb], n_results=k)
    # Always return a list of texts (may be empty)
    return results.get("documents", [[]])[0]  

# ─── Ollama + MCP setup ────────────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME  = "llama2:13b"

NEWS_MCP    = {"host": "localhost", "port": 8023}
WEATHER_MCP = {"host": "localhost", "port": 8024}

adapter_news    = OllamaMCPOAdapter(**NEWS_MCP)
adapter_weather = OllamaMCPOAdapter(**WEATHER_MCP)
ALL_TOOLS       = adapter_news.list_tools_ollama() + adapter_weather.list_tools_ollama()

client = Client(host=OLLAMA_HOST)

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are Logistics Guru, a senior supply-chain risk manager. "
        "Use the retrieved context to ground your ISO-31000 based risk analyses. "
        "Assess financial, geopolitical, and ESG factors, and recommend mitigations."
    )
}

def rag_chat(user_message: str) -> str:
    # 1) Retrieve top-k context chunks
    context_chunks = retrieve_context(user_message)
    context_block  = "\n\n".join(context_chunks)

    # 2) Build chat messages
    messages = [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {user_message}"
        }
    ]

    # 3) Call Ollama with MCP tools
    resp = client.chat(
        model=MODEL_NAME,
        messages=messages,
        tools=ALL_TOOLS
    )

    # 4) Base assistant reply
    out = resp.message.content or ""

    # 5) Dispatch any tool calls
    if getattr(resp.message, "tool_calls", None):
        news_out    = adapter_news.call_tools_from_response(resp.message.tool_calls)
        weather_out = adapter_weather.call_tools_from_response(resp.message.tool_calls)
        out        += "\n\n" + "\n".join(news_out + weather_out)

    return out

# ─── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("## 🚢 Supply-Chain Expert Chat (RAG + MCP)")
    chatbot = gr.Chatbot(type="messages")
    txt     = gr.Textbox(placeholder="Ask about supplier risk…", show_label=False)

    def respond(message, history):
        history = history or []
        # Append user turn
        history.append({"role": "user",    "content": message})
        # Generate and append assistant turn
        reply = rag_chat(message)
        history.append({"role": "assistant","content": reply})
        return history, ""  # clear input

    txt.submit(respond, [txt, chatbot], [chatbot, txt])
    gr.Button("Clear").click(lambda: None, None, chatbot)

if __name__ == "__main__":
    demo.launch()
