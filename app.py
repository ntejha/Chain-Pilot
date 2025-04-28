import os
import gradio as gr
import logging
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
from matplotlib.patches import Wedge

from ollama import Client
from ollama_mcpo_adapter import OllamaMCPOAdapter

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── RAG bootstrap ───────────────────────────────────────────────────────────
from rag.chunker import run as do_chunk
from rag.embedder import run as do_embed
from rag.indexer import run as do_index

do_chunk()
do_embed()
do_index()

# ─── Retriever setup ──────────────────────────────────────────────────────────
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

EMB_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMB_MODEL)
STORE_DIR = os.path.join(os.path.dirname(__file__), ".chromastore")
client_chroma = PersistentClient(path=STORE_DIR)
collection = client_chroma.get_collection(name="supply_chain")

def retrieve_context(query: str, k: int = 3):
    try:
        emb = model.encode(query).tolist()
        res = collection.query(query_embeddings=[emb], n_results=k)
        docs = res.get("documents", [[]])[0]
        return docs if isinstance(docs, list) else list(docs)
    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return []

# ─── MCP setup ─────────────────────────────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama2:13b")
adapter_news = OllamaMCPOAdapter(host=os.getenv("NEWS_HOST","localhost"), port=int(os.getenv("NEWS_PORT",8023)))
adapter_weather = OllamaMCPOAdapter(host=os.getenv("WEATHER_HOST","localhost"), port=int(os.getenv("WEATHER_PORT",8024)))
ALL_TOOLS = adapter_news.list_tools_ollama() + adapter_weather.list_tools_ollama()
client = Client(host=OLLAMA_HOST)

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are Logistics Guru. Use RAG context and News/Weather tools for ISO-31000 risk assessments."
}

# ─── Risk heuristics ────────────────────────────────────────────────────────────
def analyze_risk_stance(news_items, weather_items):
    news_score = sum(2 for n in (news_items or []) if any(t in str(n).lower() for t in ['delay','strike','incident','recall','disruption','accident']))
    weather_score = sum(2 for w in (weather_items or []) if any(t in str(w).lower() for t in ['storm','flood','hurricane','tornado','extreme'])) + \
                    sum(1 for w in (weather_items or []) if any(t in str(w).lower() for t in ['rain','snow','wind']))
    total = news_score + weather_score
    if total >= 5: stance = 'High Risk'
    elif total >= 2: stance = 'Medium Risk'
    else: stance = 'Low Risk'
    return stance, total

# ─── Ollama chat + tools ───────────────────────────────────────────────────────
def rag_chat_and_tools(query: str):
    if not query.strip():
        return "Please enter a company and address.", [], []
    try:
        ctx = retrieve_context(query)
        # Build user message
        ctx = retrieve_context(query)
        content = "Context:\n" + "\n\n".join(ctx) + "\n\nQuestion: " + query
        user_prompt = {"role": "user", "content": content}
        resp = client.chat(model=MODEL_NAME, messages=[SYSTEM_PROMPT, user_prompt], tools=ALL_TOOLS)
        reply = resp.message.content or ""
        calls = getattr(resp.message, 'tool_calls', [])
        news = adapter_news.call_tools_from_response(calls) or []
        weather = adapter_weather.call_tools_from_response(calls) or []
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return "Error fetching response.", [], []
    return reply, news, weather

# ─── Speedometer gauge generator ──────────────────────────────────────────────
def make_speedometer(score):
    max_score = 10
    ratio = min(score, max_score) / max_score
    fig, ax = plt.subplots(figsize=(4,2))
    # Base semicircle
    wedge = Wedge((0,0), 1, 0, 180, facecolor='#eee', edgecolor='black')
    ax.add_patch(wedge)
    # Needle
    angle = np.pi * ratio
    x, y = np.cos(angle), np.sin(angle)
    ax.plot([0, x], [0, y], lw=3, color='red')
    # Ticks
    for val in range(0, max_score+1, max_score//5):
        ang = np.pi * (val/max_score)
        ax.text(np.cos(ang)*1.1, np.sin(ang)*1.1, str(val), ha='center', va='center')
    ax.set_aspect('equal')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ─── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("## 🚢 Supply-Chain Risk Chat with Gauge Dashboard")
    with gr.Row():
        with gr.Column(scale=2, min_width=400):
            chat = gr.Chatbot(type='tuples', label='Chat')
            inp = gr.Textbox(placeholder='Company, Address...', show_label=False)
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Risk Gauge")
            gauge = gr.Image(type='pil', label='Risk Level')
            txt_out = gr.Markdown("_Awaiting input..._")

    def respond(q, hist):
        hist = hist or []
        reply, news, weather = rag_chat_and_tools(q)
        hist.append((q, reply))
        stance, total = analyze_risk_stance(news, weather)
        txt = f"**Risk Stance:** {stance} (Score: {total})"
        img = make_speedometer(total)
        return hist, img, txt

    inp.submit(respond, [inp, chat], [chat, gauge, txt_out])
    gr.Button('Clear').click(lambda: ([], None, "_Awaiting input..._"), None, [chat, gauge, txt_out])

if __name__ == '__main__':
    demo.launch()
