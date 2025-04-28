# news_api.py

from fastapi import FastAPI, HTTPException
from fastapi_mcp import FastApiMCP
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from dotenv import load_dotenv

load_dotenv("/home/tejhanagarajan/Projects/Chain-Pilot/.env")
import os
app = FastAPI(
    title="NewsAPI MCP Service",
    description="Expose NewsAPI.org endpoints as MCP tools"
)

# ————— Configure NewsAPI client —————
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))  # Obtain from https://newsapi.org/register

# ————— Top Headlines Endpoint —————
@app.get("/news/top-headlines", operation_id="get_top_headlines")
async def get_top_headlines(
    country: str = "us",
    category: str = None,
    q: str = None,
    page_size: int = 20
):
    """
    Fetch top headlines.
    """
    try:
        return newsapi.get_top_headlines(
            country=country,
            category=category,
            q=q,
            page_size=page_size
        )
    except NewsAPIException as e:
        raise HTTPException(status_code=500, detail=f"NewsAPI error: {e}")

# ————— Everything Endpoint —————
@app.get("/news/everything", operation_id="get_everything")
async def get_everything(
    q: str,
    from_param: str = None,
    to: str = None,
    language: str = "en",
    sort_by: str = "publishedAt",
    page_size: int = 20
):
    """
    Search through millions of articles.
    """
    try:
        return newsapi.get_everything(
            q=q,
            from_param=from_param,
            to=to,
            language=language,
            sort_by=sort_by,
            page_size=page_size
        )
    except NewsAPIException as e:
        raise HTTPException(status_code=500, detail=f"NewsAPI error: {e}")

# ————— Sources Endpoint —————
@app.get("/news/sources", operation_id="get_sources")
async def get_sources(
    category: str = None,
    language: str = None,
    country: str = None
):
    """
    List the news sources you can query.
    """
    try:
        return newsapi.get_sources(
            category=category,
            language=language,
            country=country
        )
    except NewsAPIException as e:
        raise HTTPException(status_code=500, detail=f"NewsAPI error: {e}")

# ————— Mount the MCP Server —————
mcp = FastApiMCP(
    app,
    name="NewsAPI MCP",
    description="Expose NewsAPI.org endpoints as MCP tools"
)
mcp.mount()  # mounts under /mcp by default

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8024)
