# Chain‑Pilot

A Gradio‑based chatbot that integrates an Ollama LLM with two FastAPI‑MCP tool servers (News and Weather) to fetch real‑time data on demand.

---

## 📁 Project Structure

```
Chain-Pilot/
├── app.py                # Main Gradio+OllamaMCPOAdapter chat application
├── mcp-server/
│   ├── news_api.py       # FastAPI-MCP server for news tool (port 8023)
│   └── weather_api.py    # FastAPI-MCP server for weather tool (port 8024)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## ⚙️ Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running locally (HTTP API on `localhost:11434`).
3. **Git** (to clone this repo).

---

## 🛠️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/Chain-Pilot.git
   cd Chain-Pilot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install Python dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Pull your Ollama model**
   ```bash
   # Example: llama2:13b or deepseek:8b
   ollama pull llama2:13b
   ```

5. **(Optional)** Edit `app.py` to change the model or MCP ports:
   ```diff
   - MODEL_NAME = "llama2:13b"
   + MODEL_NAME = "deepseek:8b"

   - NEWS_MCP = {"host": "localhost", "port": 8023}
   + NEWS_MCP = {"host": "127.0.0.1", "port": 8023}
   ```

---

## 🚀 Running the MCP Servers

Open two separate terminals (with your virtual environment activated) and start each service:

```bash
# Terminal 1 – News API
python mcp-server/news_api.py

# Terminal 2 – Weather API
python mcp-server/weather_api.py
```

Each will print FastAPI/Uvicorn startup logs and begin listening on ports 8023 and 8024 respectively.

---

## ▶️ Launching the Chat App

With the MCP servers running, start the Gradio interface:

```bash
python app.py
```

Gradio will print a local URL, e.g.:  
```
Running on http://127.0.0.1:7860
```

Open that URL in your browser to interact:

- Type questions like **“What are today’s top headlines?”** or **“What’s the weather in London?”**
- The Ollama model will invoke the News and Weather MCP tools as needed and display live data.

---

## 🌐 Environment Variables

If your Ollama API host or port differs, set the `OLLAMA_HOST` environment variable before running:

```bash
export OLLAMA_HOST=http://localhost:11434
```  
(Or adjust it at the top of `app.py`.)

---

## 🧩 Customization

- **Switch LLM model**: Edit `MODEL_NAME` at the top of `app.py` and rerun `ollama pull` if needed.
- **Add more MCP tools**: Create new FastAPI-MCP services, then update `app.py` to include a new `OllamaMCPOAdapter` and merge its `list_tools_ollama()` into `ALL_TOOLS`.

---

## 📝 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

