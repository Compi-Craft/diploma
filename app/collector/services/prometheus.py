import httpx
from ..config import settings

async def fetch_metric(query: str) -> float | None:
    url = f"{settings.PROMETHEUS_URL}/api/v1/query"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"query": query}, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", {}).get("result", [])
                if results:
                    return float(results[0]["value"][1])
            return None
    except Exception as e:
        print(f"⚠️ Prometheus Connection Error: {e}")
        return None
