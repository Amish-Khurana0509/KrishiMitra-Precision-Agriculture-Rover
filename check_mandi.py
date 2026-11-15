"""Check connectivity to the keyless Mandi API.

Run from the project folder:
    python check_mandi.py

No API key is required. This checks the states endpoint and a sample state price feed.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
base = os.getenv("MANDI_API_BASE", "https://mandi-api.onrender.com/v1").rstrip("/")
timeout = (10, 45)

def check(endpoint, params=None):
    url = f"{base}/{endpoint.lstrip('/')}"
    print(f"\nGET {url}" + (f"  params={params}" if params else ""))
    try:
        response = requests.get(url, params=params or {}, timeout=timeout)
        print("HTTP:", response.status_code)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("success") is False:
            print("API reported failure:", payload.get("error") or payload.get("message"))
            return None
        data = payload if isinstance(payload, list) else payload.get("data", [])
        if isinstance(data, dict):
            data = data.get("records", data.get("items", []))
        print("Records:", len(data) if isinstance(data, list) else "unexpected response shape")
        if isinstance(data, list) and data:
            print("Sample:", data[0])
        return data
    except requests.RequestException as exc:
        print("Request failed:", exc)
    except ValueError:
        print("The API response was not valid JSON.")
    return None

states = check("states")
if isinstance(states, list) and states:
    first = states[0]
    if isinstance(first, dict):
        first = first.get("state") or first.get("name")
    if first:
        check("prices", {"state": first})
else:
    print("\nCould not retrieve the supported-state list.")
