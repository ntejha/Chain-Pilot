import os
import gradio as gr
import time
import numpy as np
import pandas as pd
from datetime import datetime
from ollama import Client
from ollama_mcpo_adapter import OllamaMCPOAdapter
from threading import Thread
import json
import requests

# ─── Bootstrap RAG artifacts on first run ──────────────────────────────────────
from rag.chunker import run as do_chunk
from rag.embedder import run as do_embed
from rag.indexer import run as do_index
do_chunk()
do_embed()
do_index()

# ─── Retriever setup ───────────────────────────────────────────────────────────
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
EMB_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMB_MODEL)
STORE_DIR = os.path.join(os.path.dirname(__file__), ".chromastore")
client_chroma = PersistentClient(path=STORE_DIR)
collection = client_chroma.get_collection(name="supply_chain")

def retrieve_context(query: str, k: int = 3):
    query_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=k)
    # Always return a list of texts (may be empty)
    return results.get("documents", [[]])[0]

# ─── Config and Data Management ───────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
ALERTS_PATH = os.path.join(os.path.dirname(__file__), "alerts.json")

# Default configuration
DEFAULT_CONFIG = {
    "api_sources": {
        "shipping_data": "https://api.example.com/shipping",
        "inventory_data": "https://api.example.com/inventory",
        "supplier_status": "https://api.example.com/suppliers"
    },
    "alert_thresholds": {
        "shipping_delay": 3,  # Days
        "inventory_level": 15,  # Percent
        "supplier_risk": 7.5   # Score out of 10
    },
    "refresh_interval": 300,  # Seconds
    "last_updated": None
}

# Initialize config if not exists
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)

# Initialize alerts if not exists
if not os.path.exists(ALERTS_PATH):
    with open(ALERTS_PATH, 'w') as f:
        json.dump([], f)

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

def load_alerts():
    with open(ALERTS_PATH, 'r') as f:
        return json.load(f)

def save_alerts(alerts):
    with open(ALERTS_PATH, 'w') as f:
        json.dump(alerts, f, indent=2)

def add_alert(alert):
    alerts = load_alerts()
    alerts.append({
        "timestamp": datetime.now().isoformat(),
        "message": alert,
        "acknowledged": False
    })
    save_alerts(alerts)

# ─── Live Data Fetcher ───────────────────────────────────────────────────────
class LiveDataManager:
    def __init__(self):
        self.supply_data = {
            "risk_score": 3.5,
            "shipping_delays": [],
            "inventory_alerts": [],
            "supplier_status": [],
            "last_updated": None
        }
        self.running = False
        self.thread = None
    
    def start(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self.update_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def fetch_live_data(self):
        config = load_config()
        api_sources = config["api_sources"]
        
        # In a real implementation, these would be actual API calls
        # For demo, we'll simulate with random data
        try:
            # Simulated shipping data
            shipping_delays = []
            for i in range(np.random.randint(0, 4)):
                delay_days = np.random.randint(1, 10)
                shipping_delays.append({
                    "shipment_id": f"SH-{np.random.randint(10000, 99999)}",
                    "route": f"{np.random.choice(['Shanghai', 'Rotterdam', 'Singapore'])} to {np.random.choice(['LA', 'NY', 'Hamburg'])}",
                    "delay_days": delay_days,
                    "reason": np.random.choice(["Weather", "Port congestion", "Customs delay", "Mechanical issues"])
                })
                
                # Check for alert threshold
                if delay_days > config["alert_thresholds"]["shipping_delay"]:
                    add_alert(f"ALERT: Shipment {shipping_delays[-1]['shipment_id']} delayed by {delay_days} days due to {shipping_delays[-1]['reason']}")
            
            # Simulated inventory data
            inventory_alerts = []
            for i in range(np.random.randint(0, 3)):
                inventory_pct = np.random.randint(5, 30)
                inventory_alerts.append({
                    "product_id": f"P-{np.random.randint(1000, 9999)}",
                    "product_name": np.random.choice(["Microchips", "Batteries", "Circuit boards", "Displays", "Connectors"]),
                    "inventory_percent": inventory_pct,
                    "restock_eta": f"{np.random.randint(1, 14)} days"
                })
                
                # Check for alert threshold
                if inventory_pct < config["alert_thresholds"]["inventory_level"]:
                    add_alert(f"ALERT: Low inventory for {inventory_alerts[-1]['product_name']} ({inventory_pct}%) - Restock ETA: {inventory_alerts[-1]['restock_eta']}")
            
            # Simulated supplier status
            supplier_status = []
            for i in range(np.random.randint(1, 5)):
                risk_score = round(np.random.uniform(1, 10), 1)
                supplier_status.append({
                    "supplier_id": f"SUP-{np.random.randint(100, 999)}",
                    "supplier_name": np.random.choice(["TechSystems Inc.", "GlobalParts Ltd.", "ElectroSupply Co.", "AsiaComponents", "EuroPrecision"]),
                    "risk_score": risk_score,
                    "risk_factors": np.random.choice(["Financial instability", "Geopolitical issues", "Labor strikes", "Quality concerns", "Capacity constraints"], 
                                                   size=np.random.randint(0, 3), replace=False).tolist()
                })
                
                # Check for alert threshold
                if risk_score > config["alert_thresholds"]["supplier_risk"]:
                    add_alert(f"ALERT: High risk score ({risk_score}/10) for supplier {supplier_status[-1]['supplier_name']} - Factors: {', '.join(supplier_status[-1]['risk_factors'])}")
            
            # Calculate overall risk score (weighted average of various factors)
            risk_components = [
                3.0,  # Base risk
                np.mean([d["delay_days"] for d in shipping_delays]) / 2 if shipping_delays else 0,
                np.mean([10 - i["inventory_percent"]/10 for i in inventory_alerts]) if inventory_alerts else 0,
                np.mean([s["risk_score"]/2 for s in supplier_status]) if supplier_status else 0
            ]
            risk_score = min(10, max(1, sum(risk_components) / len(risk_components)))
            
            self.supply_data = {
                "risk_score": round(risk_score, 1),
                "shipping_delays": shipping_delays,
                "inventory_alerts": inventory_alerts,
                "supplier_status": supplier_status,
                "last_updated": datetime.now().isoformat()
            }
            
            # Update config with last updated time
            config["last_updated"] = self.supply_data["last_updated"]
            save_config(config)
            
            return True
        except Exception as e:
            print(f"Error fetching live data: {e}")
            return False
    
    def update_loop(self):
        while self.running:
            self.fetch_live_data()
            config = load_config()
            time.sleep(config["refresh_interval"])
    
    def get_data(self):
        return self.supply_data
    
    def get_risk_data(self):
        return {
            "score": self.supply_data["risk_score"],
            "level": self.get_risk_level(),
            "last_updated": self.supply_data["last_updated"]
        }
    
    def get_risk_level(self):
        score = self.supply_data["risk_score"]
        if score < 3.5:
            return "Low"
        elif score < 7:
            return "Medium"
        else:
            return "High"
    
    def get_summary(self):
        data = self.supply_data
        summary = []
        
        if data["shipping_delays"]:
            summary.append(f"🚢 {len(data['shipping_delays'])} active shipping delays (avg {np.mean([d['delay_days'] for d in data['shipping_delays']]):.1f} days)")
        
        if data["inventory_alerts"]:
            summary.append(f"📦 {len(data['inventory_alerts'])} inventory alerts (avg {np.mean([i['inventory_percent'] for i in data['inventory_alerts']]):.1f}% remaining)")
        
        if data["supplier_status"]:
            high_risk = [s for s in data["supplier_status"] if s["risk_score"] > 7]
            if high_risk:
                summary.append(f"⚠️ {len(high_risk)} high-risk suppliers (out of {len(data['supplier_status'])})")
        
        return "\n".join(summary) if summary else "No active supply chain issues detected."

# Initialize live data manager
live_data = LiveDataManager()
live_data.start()

# ─── Ollama + MCP setup ────────────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama2:13b"
NEWS_MCP = {"host": "localhost", "port": 8023}
WEATHER_MCP = {"host": "localhost", "port": 8024}
adapter_news = OllamaMCPOAdapter(**NEWS_MCP)
adapter_weather = OllamaMCPOAdapter(**WEATHER_MCP)
ALL_TOOLS = adapter_news.list_tools_ollama() + adapter_weather.list_tools_ollama()
client = Client(host=OLLAMA_HOST)

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are Logistics Guru, a senior supply-chain risk manager. "
        "Use the retrieved context and real-time supply chain data to ground your ISO-31000 based risk analyses. "
        "When answering, first check if there are any real-time alerts or data points relevant to the question. "
        "Assess financial, geopolitical, and ESG factors, and recommend mitigations."
    )
}

def rag_chat(user_message: str) -> str:
    # 1) Retrieve top-k context chunks
    context_chunks = retrieve_context(user_message)
    context_block = "\n\n".join(context_chunks)
    
    # 2) Get real-time supply chain data
    supply_data = live_data.get_data()
    data_summary = live_data.get_summary()
    
    # 3) Build chat messages with both RAG context and real-time data
    messages = [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": (
                f"Context:\n{context_block}\n\n"
                f"Real-time Supply Chain Data ({datetime.now().strftime('%Y-%m-%d %H:%M')}):\n"
                f"Overall Risk Score: {supply_data['risk_score']}/10 ({live_data.get_risk_level()} Risk)\n"
                f"{data_summary}\n\n"
                f"Question: {user_message}"
            )
        }
    ]
    
    # 4) Call Ollama with MCP tools
    resp = client.chat(
        model=MODEL_NAME,
        messages=messages,
        tools=ALL_TOOLS
    )
    
    # 5) Base assistant reply
    out = resp.message.content or ""
    
    # 6) Dispatch any tool calls
    if getattr(resp.message, "tool_calls", None):
        news_out = adapter_news.call_tools_from_response(resp.message.tool_calls)
        weather_out = adapter_weather.call_tools_from_response(resp.message.tool_calls)
        out += "\n\n" + "\n".join(news_out + weather_out)
    
    return out

# ─── Admin Functions ────────────────────────────────────────────────────────────
def update_api_source(name, url):
    config = load_config()
    config["api_sources"][name] = url
    save_config(config)
    return f"API source '{name}' updated to {url}"

def update_alert_threshold(name, value):
    config = load_config()
    try:
        value = float(value)
        config["alert_thresholds"][name] = value
        save_config(config)
        return f"Alert threshold '{name}' updated to {value}"
    except ValueError:
        return "Error: Threshold must be a number"

def update_refresh_interval(seconds):
    config = load_config()
    try:
        seconds = int(seconds)
        if seconds < 60:
            return "Error: Refresh interval must be at least 60 seconds"
        config["refresh_interval"] = seconds
        save_config(config)
        return f"Refresh interval updated to {seconds} seconds"
    except ValueError:
        return "Error: Interval must be a number"

def refresh_data():
    success = live_data.fetch_live_data()
    return "Data refreshed successfully" if success else "Error refreshing data"

def get_alerts():
    alerts = load_alerts()
    # Sort by timestamp (newest first) and limit to most recent 20
    return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)[:20]

def acknowledge_alert(index):
    alerts = load_alerts()
    if 0 <= index < len(alerts):
        alerts[index]["acknowledged"] = True
        save_alerts(alerts)
        return f"Alert {index+1} acknowledged"
    return f"Invalid alert index: {index}"

# ─── Risk Meter Visualization (replaced Gauge) ────────────────────────────────
def create_risk_meter():
    risk_data = live_data.get_risk_data()
    score = risk_data["score"]
    level = risk_data["level"]
    
    # Create HTML for risk meter visualization
    color = "green" if score < 3.5 else "orange" if score < 7 else "red"
    
    # Format the risk level as text
    risk_info = f"{level} Risk ({score}/10)"
    last_updated = "Never" if not risk_data["last_updated"] else datetime.fromisoformat(risk_data["last_updated"]).strftime("%H:%M:%S")
    
    # Create a text visualization instead of a gauge
    meter_html = f"""
    <div style="text-align: center; margin-bottom: 10px;">
        <h3>Supply Chain Risk Level</h3>
        <div style="margin: 0 auto; width: 80%; height: 30px; background-color: #eee; border-radius: 15px; overflow: hidden;">
            <div style="width: {score*10}%; height: 100%; background-color: {color};"></div>
        </div>
        <p style="margin-top: 5px; font-weight: bold; color: {color};">{risk_info}</p>
        <p style="margin-top: 2px; font-size: 0.8em;">Last updated: {last_updated}</p>
    </div>
    """
    
    return meter_html, risk_info

def format_alerts_for_display():
    alerts = get_alerts()
    if not alerts:
        return "No recent alerts"
    
    formatted = []
    for i, alert in enumerate(alerts):
        status = "✓" if alert["acknowledged"] else "!"
        timestamp = datetime.fromisoformat(alert["timestamp"]).strftime("%m-%d %H:%M")
        formatted.append(f"{i+1}. {status} [{timestamp}] {alert['message']}")
    
    return "\n".join(formatted)

def update_dashboard():
    risk_data = live_data.get_risk_data()
    summary = live_data.get_summary()
    alerts = format_alerts_for_display()
    
    # Create HTML for risk meter
    meter_html, risk_info = create_risk_meter()
    
    return (
        meter_html,
        risk_info,
        summary,
        alerts
    )

def ack_selected_alert(alert_index):
    try:
        index = int(alert_index) - 1  # Convert to 0-based index
        result = acknowledge_alert(index)
        return result, format_alerts_for_display()
    except ValueError:
        return "Please enter a valid alert number", format_alerts_for_display()

with gr.Blocks() as demo:
    gr.Markdown("# 🚢 Supply-Chain Expert System (RAG + MCP + Real-time Data)")
    
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=500)
        txt = gr.Textbox(placeholder="Ask about supplier risk, shipping delays, inventory levels...", show_label=False)
        
        with gr.Row():
            clear_btn = gr.Button("Clear Chat")
            refresh_btn = gr.Button("Refresh Data")
        
        with gr.Accordion("Live Supply Chain Status", open=False):
            risk_meter_html = gr.HTML()
            risk_level = gr.Textbox(label="Risk Level", interactive=False)
            status_summary = gr.Textbox(label="Current Issues", value=live_data.get_summary(), lines=3, interactive=False)
            alerts_box = gr.Textbox(label="Recent Alerts", value=format_alerts_for_display(), lines=5, interactive=False)
            
            with gr.Row():
                alert_index = gr.Textbox(label="Alert #", value="", placeholder="Enter alert number")
                ack_btn = gr.Button("Acknowledge Alert")
    
    with gr.Tab("Dashboard"):
        gr.Markdown("## Supply Chain Risk Dashboard")
        
        dashboard_risk_meter = gr.HTML()
        dashboard_risk_info = gr.Textbox(label="Risk Status", value="", interactive=False)
        
        with gr.Row():
            dashboard_summary = gr.Textbox(label="Supply Chain Summary", value=live_data.get_summary(), lines=5, interactive=False)
            dashboard_alerts = gr.Textbox(label="Active Alerts", value=format_alerts_for_display(), lines=8, interactive=False)
        
        dashboard_refresh_btn = gr.Button("Refresh Dashboard")
        auto_refresh_note = gr.Markdown("**Note:** Click the Refresh Dashboard button periodically to see the latest data.")
    
    with gr.Tab("Admin"):
        gr.Markdown("## Data Source Management")
        
        with gr.Row():
            with gr.Column():
                api_name = gr.Textbox(label="API Name", placeholder="e.g., shipping_data")
                api_url = gr.Textbox(label="API URL", placeholder="https://api.example.com/endpoint")
                update_api_btn = gr.Button("Update API Source")
                api_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column():
                threshold_name = gr.Dropdown(
                    label="Alert Threshold", 
                    choices=["shipping_delay", "inventory_level", "supplier_risk"],
                    value="shipping_delay"
                )
                threshold_value = gr.Number(label="Threshold Value")
                update_threshold_btn = gr.Button("Update Threshold")
                threshold_status = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("## System Settings")
        with gr.Row():
            refresh_interval = gr.Number(label="Data Refresh Interval (seconds)", value=300, minimum=60)
            update_interval_btn = gr.Button("Update Interval")
            interval_status = gr.Textbox(label="Status", interactive=False)
    
    # Initialize the dashboard elements
    meter_html, risk_info = create_risk_meter()
    risk_meter_html.value = meter_html
    risk_level.value = risk_info
    dashboard_risk_meter.value = meter_html
    dashboard_risk_info.value = risk_info
    
    # Fixed respond function for the chatbot
    def respond(message, history):
        # Generate reply using RAG chat
        reply = rag_chat(message)
        
        # Update risk meter
        meter_html, risk_info = create_risk_meter()
        
        # Return the updated history by appending the new exchange as a list of pairs
        return history + [[message, reply]], "", meter_html, risk_info, live_data.get_summary(), format_alerts_for_display()
    
    # Chat tab events
    txt.submit(respond, [txt, chatbot], [chatbot, txt, risk_meter_html, risk_level, status_summary, alerts_box])
    # Fix for clear button - return empty list instead of None
    clear_btn.click(lambda: [], None, chatbot)
    
    # Handle refresh button click
    def on_refresh():
        refresh_data()
        meter_html, risk_info = create_risk_meter()
        return meter_html, risk_info, live_data.get_summary(), format_alerts_for_display()
    
    refresh_btn.click(
        on_refresh,
        None, 
        [risk_meter_html, risk_level, status_summary, alerts_box]
    )
    
    ack_btn.click(
        ack_selected_alert,
        [alert_index],
        [status_summary, alerts_box]
    )
    
    # Dashboard tab events
    def update_dashboard_ui():
        refresh_data()  # Force a data refresh
        meter_html, risk_info = create_risk_meter()
        return meter_html, risk_info, live_data.get_summary(), format_alerts_for_display()
    
    dashboard_refresh_btn.click(
        update_dashboard_ui,
        None,
        [dashboard_risk_meter, dashboard_risk_info, dashboard_summary, dashboard_alerts]
    )
    
    # Admin tab events
    update_api_btn.click(
        update_api_source,
        [api_name, api_url],
        [api_status]
    )
    
    update_threshold_btn.click(
        update_alert_threshold,
        [threshold_name, threshold_value],
        [threshold_status]
    )
    
    update_interval_btn.click(
        update_refresh_interval,
        [refresh_interval],
        [interval_status]
    )

if __name__ == "__main__":
    # Make sure the data is loaded before launching
    live_data.fetch_live_data()
    demo.launch()