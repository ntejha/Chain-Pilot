# weather_api.py

from fastapi import FastAPI, HTTPException
from fastapi_mcp import FastApiMCP
import weatherapi
from weatherapi.rest import ApiException
from dotenv import load_dotenv
import os

load_dotenv("/home/tejhanagarajan/Projects/Chain-Pilot/.env")

app = FastAPI(
    title="WeatherAPI MCP Service",
    description="Expose WeatherAPI.com’s current and forecast endpoints as MCP tools"
)

# ————— Configure WeatherAPI client —————
configuration = weatherapi.Configuration()
configuration.api_key['key'] = os.getenv("WEATHER_API_KEY")  # Obtain from https://www.weatherapi.com/account
api_client = weatherapi.ApiClient(configuration)
api_instance = weatherapi.APIsApi(api_client)

# ————— Current Weather Endpoint —————
@app.get("/weather/current", operation_id="get_current_weather")
async def get_current_weather(city: str):
    """
    Fetch real-time weather for the specified city.
    """
    try:
        return api_instance.realtime_weather(q=city)
    except ApiException as e:
        raise HTTPException(status_code=500, detail=f"WeatherAPI error: {e}")

# ————— Forecast Endpoint —————
@app.get("/weather/forecast", operation_id="get_weather_forecast")
async def get_weather_forecast(city: str, days: int = 3):
    """
    Fetch a forecast for the next `days` days (1–14).
    """
    try:
        return api_instance.forecast_weather(q=city, days=days, aqi=False, alerts=False)
    except ApiException as e:
        raise HTTPException(status_code=500, detail=f"WeatherAPI error: {e}")

# ————— Mount the MCP Server —————
mcp = FastApiMCP(
    app,
    name="WeatherAPI MCP",
    description="Expose WeatherAPI.com endpoints as MCP tools"
)
mcp.mount()  # defaults to mounting under /mcp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8023)
