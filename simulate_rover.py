"""Send fake rover telemetry to a running KrishiMitra server, to test the Rover page.

    python simulate_rover.py                       # http://127.0.0.1:5000
    python simulate_rover.py http://192.168.1.20:5000
Reads TELEMETRY_TOKEN from .env if set. Stop with Ctrl+C. Uses only requests + python-dotenv.
"""
import os, sys, time, math
import requests
from dotenv import load_dotenv

load_dotenv()
base = (sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:5000').rstrip('/')
headers = {'X-Device-Token': os.getenv('TELEMETRY_TOKEN', '')} if os.getenv('TELEMETRY_TOKEN') else {}
start = time.time()
while True:
    t = time.time() - start
    payload = {'battery_percent': round(max(5, 90 - t / 20), 1), 'battery_voltage': 12.4, 'tank_a_percent': 90, 'tank_b_percent': 65,
               'tank_c_percent': round(max(0, 50 - t / 30), 1), 'latitude': 28.61 + math.sin(t / 30) * 1e-4, 'longitude': 77.20 + math.cos(t / 30) * 1e-4,
               'speed_kmh': 0.4, 'area_ha': round(t / 5000, 4), 'runtime_seconds': int(t), 'status': 'Test packet'}
    try:
        r = requests.post(base + '/api/telemetry', json=payload, headers=headers, timeout=5)
        print(r.status_code, r.text.strip())
    except requests.RequestException as e:
        print('Failed:', e)
    time.sleep(3)
