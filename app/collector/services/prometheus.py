import httpx
from shared.logger import send_system_log

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=4.0,
            limits=httpx.Limits(max_keepalive_connections=5, keepalive_expiry=30),
        )
    return _client


async def fetch_metric(query: str, prom_url: str) -> float | None:
    try:
        client = _get_client()
        response = await client.get(prom_url, params={"query": query})
        if response.status_code == 200:
            data = response.json()
            results = data.get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1])
        return None
    except Exception:
        return None
