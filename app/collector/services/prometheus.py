import httpx
from ..config import settings

async def fetch_metric(query: str, prom_url: str) -> float | None:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(prom_url, params={"query": query}, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", {}).get("result", [])
                if results:
                    return float(results[0]["value"][1])
            return None
    except Exception as e:
        print(f"⚠️ Prometheus Connection Error: {e}")
        return None
