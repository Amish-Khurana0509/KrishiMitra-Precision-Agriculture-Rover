from __future__ import annotations
import base64, io, json, os, re, time, threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, url_for
from markupsafe import Markup, escape
from werkzeug.utils import secure_filename

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config.update(SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'change-this-before-deployment'), MAX_CONTENT_LENGTH=10 * 1024 * 1024)

# ---------------------------------------------------------------- configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '').strip()
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'nvidia/nemotron-3.5-super-vl:free').strip()
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
TELEMETRY_TOKEN = os.getenv('TELEMETRY_TOKEN', '').strip()
AI_CONFIGURED = bool(OPENROUTER_API_KEY)

LANGUAGES = ['English', 'Hindi', 'Marathi', 'Gujarati', 'Punjabi', 'Tamil', 'Telugu', 'Bengali']
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

_telemetry_lock = threading.Lock()
_latest_telemetry: dict[str, Any] = {'connected': False, 'received_at': None, 'source': 'No device data received'}
_last_telemetry_monotonic = 0.0
_model = None
_model_error = None
_model_lock = threading.Lock()

DISEASE_NOTES = {
    'bacterial spot': 'Small, dark, water-soaked spots that can spread in warm, wet weather. Avoid working among wet plants, remove badly affected leaves where practical, use clean seed/seedlings and rotate crops. Confirm with a local KVK before any treatment.',
    'early blight': 'Brown spots with concentric rings, usually on older leaves first. Remove badly affected leaves, avoid overhead watering, mulch to limit soil splash and improve airflow. Confirm the diagnosis with a local KVK or plant pathologist.',
    'late blight': 'Large, irregular dark patches that can spread very quickly in cool, humid weather. Isolate affected material where practical, avoid moving wet foliage between plants, and seek prompt local agricultural advice.',
    'leaf mold': 'Pale patches on the upper leaf surface with mould underneath, common in humid, poorly ventilated crops. Improve ventilation, avoid wetting leaves and monitor nearby plants. Confirm before treatment.',
    'septoria': 'Many small circular spots with dark edges and pale centres, mostly on lower leaves. Remove affected lower leaves, avoid overhead watering, mulch and rotate crops. Confirm with a local expert.',
    'spider mite': 'Fine stippling and webbing, usually on leaf undersides in hot, dry conditions. Check the underside of leaves with a magnifier, reduce dust and water stress, and ask your KVK about safe control options.',
    'target spot': 'Brown spots with target-like rings on leaves and sometimes fruit, favoured by warm, humid weather. Improve airflow, avoid wet foliage and remove infected debris. Confirm the diagnosis locally.',
    'yellow leaf curl': 'Upward-curling, yellowing and stunted leaves, spread by whiteflies. Control whitefly numbers (yellow sticky traps, nets in the nursery), remove infected plants early and ask your KVK about resistant varieties.',
    'mosaic': 'Mottled light/dark green leaves and distortion; the virus spreads by contact and contaminated tools or seed. Remove infected plants, wash hands and tools, and do not handle healthy plants after infected ones.',
    'healthy': 'The model classified the visible plant tissue as healthy. Continue routine monitoring; a photo cannot rule out every plant-health issue.',
}
GENERIC_NOTE = 'The model detected a possible plant-health class. Use this as a screening aid, not a confirmed diagnosis. Check symptoms and consult a local KVK before treatment.'

_rate_hits: dict[str, deque] = defaultdict(deque)

def rate_limited(limit: int = 20, window: int = 60):
    """Tiny per-IP limiter for endpoints that spend paid API quota."""
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            now = time.monotonic(); q = _rate_hits[f'{fn.__name__}:{request.remote_addr}']
            while q and now - q[0] > window: q.popleft()
            if len(q) >= limit:
                return jsonify(error='Too many requests. Please wait a minute and try again.'), 429
            q.append(now)
            return fn(*a, **kw)
        return wrapper
    return deco

def render_markdown(text: str) -> Markup:
    """Escape first, then support the small markdown subset that AI models commonly return."""
    out, in_list = [], False
    def inline(s: str) -> str:
        s = str(escape(s))
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
        return re.sub(r'(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\w)', r'<em>\1</em>', s)
    for raw in (text or '').splitlines():
        line = raw.rstrip()
        m = re.match(r'^\s*(?:[-*•]|\d+[.)])\s+(.*)$', line)
        if m:
            if not in_list: out.append('<ul>'); in_list = True
            out.append(f'<li>{inline(m.group(1))}</li>'); continue
        if in_list: out.append('</ul>'); in_list = False
        h = re.match(r'^#{1,6}\s+(.*)$', line)
        if h: out.append(f'<h4>{inline(h.group(1))}</h4>')
        elif line.strip(): out.append(f'<p>{inline(line)}</p>')
    if in_list: out.append('</ul>')
    return Markup('\n'.join(out))

app.jinja_env.filters['md'] = render_markdown

MODEL_CLASSES = {
    0: 'Tomato Bacterial Spot',
    1: 'Tomato Early blight',
    2: 'Tomato Late blight',
    3: 'Tomato Leaf Mold',
    4: 'Tomato Septoria leaf spot',
    5: 'Tomato Spider mites Two-spotted spider mite',
    6: 'Tomato Target Spot',
    7: 'Tomato Yellow Leaf Curl Virus',
    8: 'Tomato healthy',
    9: 'Tomato mosaic virus',
}

def get_model():
    global _model, _model_error
    with _model_lock:
        if _model is not None:
            return _model
        if _model_error:
            return None
        model_path = BASE_DIR / 'model.onnx'
        if not model_path.exists():
            _model_error = 'model.onnx is missing from the project folder.'
            return None
        try:
            import onnxruntime as ort
            _model = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            return _model
        except Exception as exc:
            _model_error = f'Leaf model could not load: {exc}'
            return None

def _run_onnx_scan(session, image, conf_thresh: float = 0.20) -> list[dict]:
    import numpy as np
    from PIL import Image

    orig_w, orig_h = image.size
    size = 640
    scale = min(size / orig_w, size / orig_h)
    nw, nh = int(round(orig_w * scale)), int(round(orig_h * scale))
    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))

    tensor = np.transpose(np.asarray(canvas, dtype=np.float32) / 255.0, (2, 0, 1))[None, ...]
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: tensor})[0][0].T  # (8400, 14)

    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    cls_scores = preds[:, 4:]
    cls_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores[np.arange(len(preds)), cls_ids]

    mask = confs >= conf_thresh
    if not np.any(mask):
        return []

    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    confs, cls_ids = confs[mask], cls_ids[mask]
    x1 = np.clip((cx - bw / 2 - pad_x) / scale, 0, orig_w)
    y1 = np.clip((cy - bh / 2 - pad_y) / scale, 0, orig_h)
    x2 = np.clip((cx + bw / 2 - pad_x) / scale, 0, orig_w)
    y2 = np.clip((cy + bh / 2 - pad_y) / scale, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-6)
        order = order[1:][iou <= 0.45]

    detections = [
        {
            'label': MODEL_CLASSES.get(int(cls_ids[idx]), f'Class {int(cls_ids[idx])}'),
            'confidence': float(confs[idx]),
            'box': boxes[idx].tolist(),
        }
        for idx in keep
    ]
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections

# AI chatbots (OpenRouter nemotron)
_last_ai_error = ''

def call_openrouter(messages: list[dict], system: str) -> tuple[str | None, Any]:
    """Returns (answer, reasoning_details). Reasoning is enabled; it is passed back on follow-ups but never shown."""
    global _last_ai_error
    if not OPENROUTER_API_KEY: return None, None
    payload = {'model': OPENROUTER_MODEL, 'messages': [{'role': 'system', 'content': system}] + messages,
               'reasoning': {'enabled': True}, 'temperature': 0.35, 'max_tokens': 4000}
    headers = {'Authorization': f'Bearer {OPENROUTER_API_KEY}', 'Content-Type': 'application/json', 'X-Title': 'AgriSetu by KrishiMITRA'}
    for attempt in range(2):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
            if r.status_code in (429, 502, 503) and attempt == 0:
                time.sleep(3); continue
            r.raise_for_status()
            msg = r.json()['choices'][0]['message']
            text = (msg.get('content') or '').strip()
            if not text: _last_ai_error = 'No response was generated. Please try again.'; return None, None
            return text, msg.get('reasoning_details')
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            _last_ai_error = {401: 'Advisory service credentials could not be verified.', 402: 'Advisory service quota has been reached.',
                              404: 'Advisory service is temporarily unavailable.',
                              429: 'The advisory service is busy right now. Please wait a minute and try again.'}.get(code, f'Advisory service returned HTTP {code}.')
            app.logger.warning('AI provider HTTP %s', code); return None, None
        except requests.RequestException as exc:
            _last_ai_error = 'Could not reach the advisory service. Check your internet connection.'
            app.logger.warning('AI provider request failed: %s', exc); return None, None
        except (KeyError, IndexError, ValueError) as exc:
            _last_ai_error = 'Unexpected response from the advisory service.'
            app.logger.warning('AI provider parse failed: %s', exc); return None, None
    return None, None

SYSTEMS = {
    'general': 'You are AgriSetu by KrishiMITRA, a practical agricultural information assistant for Indian farmers. Give clear, concise, cautious, actionable advice. Ask for crop, state, season and symptoms if needed. Never invent government scheme eligibility, market prices, pesticide dosages or legal guarantees. Recommend local KVK/agriculture department confirmation for high-impact decisions.',
    'advisor': 'You are AgriSetu by KrishiMITRA, an Indian crop advisory assistant. Give stage-wise guidance and explain uncertainty. Ask for location, sowing date, variety, soil and irrigation context when needed. Do not prescribe chemical pesticide dosages. Prefer integrated pest management and local KVK confirmation.',
    'legal': 'You are AgriSetu by KrishiMITRA. Explain agricultural government documents in plain language, not as a lawyer. Only assert details supported by the supplied document text. Clearly distinguish eligibility, application steps, deadlines and caveats. Say when the document does not answer a question.',
}

def ai_reply(prompt: str, context: str = 'general', language: str = 'English', history: list[dict] | None = None) -> dict:
    system = SYSTEMS.get(context, SYSTEMS['general']) + f' Respond in {language}. Use short paragraphs or simple bullet lists; avoid tables.'
    if not OPENROUTER_API_KEY:
        return {'text': 'The advisory assistant is currently offline. Please try again later.', 'provider': 'Offline', 'ok': False, 'reasoning_details': None}
    text, reasoning = call_openrouter((history or []) + [{'role': 'user', 'content': prompt}], system)
    if text: return {'text': text, 'provider': 'AgriSetu Guide', 'ok': True, 'reasoning_details': reasoning}
    return {'text': _last_ai_error or 'The advisory service could not be reached. Please try again.', 'provider': 'Unavailable', 'ok': False, 'reasoning_details': None}

def ai_answer(prompt: str, context: str = 'general', language: str = 'English') -> tuple[str, str, bool]:
    r = ai_reply(prompt, context, language); return r['text'], r['provider'], r['ok']

# mandi data
MANDI_API_BASE = os.getenv('MANDI_API_BASE', 'https://mandi-api.onrender.com/v1').rstrip('/')
MANDI_TTL = 1800
MANDI_PER_PAGE = 50
SUPPORTED_STATES = ['Maharashtra', 'Uttar Pradesh', 'Punjab', 'Madhya Pradesh', 'Karnataka']
STATES = SUPPORTED_STATES
COMMODITIES = ['Tomato', 'Potato', 'Onion', 'Wheat', 'Rice', 'Maize', 'Mustard', 'Soyabean', 'Cotton']
MANDI_SORTS = {'newest': 'Newest first', 'high': 'Price: high to low', 'low': 'Price: low to high', 'az': 'Commodity A-Z'}
_mandi_lock = threading.Lock()
_mandi: dict[str, Any] = {
    'rows': [], 'total': 0, 'loaded_at': 0.0, 'loaded_wall': None,
    'attempt_at': -1e9, 'error': None, 'partial': False, 'thread': None,
    'states': [], 'commodities': [], 'districts': {}, 'latest': '',
}

_ALIASES = {
    'commodity': ('commodity',), 'market': ('market',), 'state': ('state',),
    'district': ('district',), 'variety': ('variety',), 'grade': ('grade',),
    'min_price': ('min_price', 'min_x0020_price'),
    'max_price': ('max_price', 'max_x0020_price'),
    'modal_price': ('modal_price', 'modal_x0020_price'),
    'arrival_date': ('arrival_date', 'date'),
}

def _field(row: dict, name: str) -> str:
    lowered = {str(k).casefold(): v for k, v in row.items()}
    for alias in _ALIASES.get(name, (name,)):
        v = lowered.get(alias)
        if v not in (None, ''):
            return str(v).strip()
    return ''

def _date_key(s: str):
    for fmt in ('%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y'):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return datetime.min

def _num(v: str) -> float | None:
    try:
        return float(str(v).replace(',', ''))
    except (TypeError, ValueError):
        return None

def _normalize(row: dict) -> dict:
    item: dict[str, Any] = {
        k: _field(row, k)
        for k in ('commodity', 'market', 'state', 'district', 'variety', 'grade', 'arrival_date')
    }
    for k in ('min_price', 'max_price', 'modal_price'):
        item[k] = _num(_field(row, k))
    item['_date'] = _date_key(item['arrival_date'])
    return item

def _public(item: dict) -> dict:
    return {k: v for k, v in item.items() if not k.startswith('_')}

def inr(v) -> str:
    """12345.0 -> '12,345' (Indian digit grouping)."""
    if v is None:
        return '—'
    s = str(int(round(abs(v))))
    if len(s) > 3:
        s = re.sub(r'(\d)(?=(\d\d)+$)', r'\1,', s[:-3]) + ',' + s[-3:]
    return ('-' if v < 0 else '') + s
app.jinja_env.filters['inr'] = inr

def _api_get(endpoint: str, params: dict | None = None):
    url = f'{MANDI_API_BASE}/{endpoint.lstrip("/")}'
    try:
        response = requests.get(url, params=params or {}, timeout=(10, 45))
        if response.status_code == 429:
            return None, 'The Mandi API rate limit was reached. Cached prices will remain available until the next refresh.'
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get('success') is False:
            return None, str(payload.get('error') or payload.get('message') or 'The Mandi API reported an unsuccessful request.')
        return payload, None
    except requests.Timeout:
        app.logger.warning('Mandi API timed out: %s', endpoint)
        return None, 'The Mandi API is taking too long to respond. It may be starting up; please retry shortly.'
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else '?'
        app.logger.warning('Mandi API HTTP %s for %s', code, endpoint)
        return None, f'The Mandi API returned HTTP {code}.'
    except requests.RequestException as exc:
        app.logger.warning('Mandi API request failed for %s: %s', endpoint, exc)
        return None, 'Could not connect to the Mandi API. Check the server connection and retry.'
    except ValueError:
        return None, 'The Mandi API returned invalid JSON.'

def _payload_data(payload):
    """Accept the documented {success, data, meta} envelope and direct arrays."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get('data', payload.get('records', payload.get('results', [])))
        if isinstance(data, dict):
            data = data.get('records', data.get('items', []))
        return data if isinstance(data, list) else []
    return []

def _load_mandi_dataset() -> None:
    """Refresh the cache using one daily-prices request per supported state."""
    rows: list[dict] = []
    errors: list[str] = []

    states_payload, states_error = _api_get('states')
    api_states = _payload_data(states_payload) if states_payload is not None else []
    states = []
    for value in api_states:
        if isinstance(value, str):
            name = value.strip()
        elif isinstance(value, dict):
            name = str(value.get('state') or value.get('name') or '').strip()
        else:
            name = ''
        if name:
            states.append(name)
    if not states:
        states = list(SUPPORTED_STATES)
        if states_error:
            app.logger.info('Mandi states endpoint unavailable; using documented states: %s', states_error)

    successful_states = 0
    for state_name in states:
        payload, error = _api_get('prices', {'state': state_name})
        if error:
            errors.append(f'{state_name}: {error}')
            continue
        records = _payload_data(payload)
        if not records:
            successful_states += 1
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            normalized = _normalize(record)
            if not normalized['state']:
                normalized['state'] = state_name
            rows.append(normalized)
        successful_states += 1

    if not successful_states and not rows:
        error = errors[0] if errors else 'The Mandi API returned no price records.'
    elif errors:
        error = 'Some state price feeds could not be refreshed. ' + ' '.join(errors[:3])
    else:
        error = None

    with _mandi_lock:
        old_count = len(_mandi['rows'])
        if rows:
            if not errors or len(rows) >= old_count or not old_count:
                _mandi.update(
                    rows=rows, total=len(rows), loaded_at=time.monotonic(),
                    loaded_wall=datetime.now(), partial=bool(errors), error=error,
                    states=sorted({r['state'] for r in rows if r['state']}, key=str.casefold)
                )
                _mandi['commodities'] = sorted(
                    {r['commodity'] for r in rows if r['commodity']}, key=str.casefold
                )
                districts: dict[str, set] = {}
                for row in rows:
                    if row['district']:
                        districts.setdefault(row['state'], set()).add(row['district'])
                _mandi['districts'] = {
                    state: sorted(values, key=str.casefold)
                    for state, values in districts.items()
                }
                latest = max((r['_date'] for r in rows), default=datetime.min)
                _mandi['latest'] = latest.strftime('%d %b %Y') if latest != datetime.min else ''
            elif error:
                _mandi['error'] = error
        else:
            _mandi['error'] = error or 'The Mandi API returned no price records.'

def ensure_mandi(wait: float = 0.0) -> None:
    now = time.monotonic()
    thread = None
    with _mandi_lock:
        current = _mandi['thread']
        running = current is not None and current.is_alive()
        fresh = bool(_mandi['loaded_at'] and now - _mandi['loaded_at'] < MANDI_TTL)
        if not running and not fresh and now - _mandi['attempt_at'] > 60:
            _mandi['attempt_at'] = now
            current = threading.Thread(
                target=_load_mandi_dataset, daemon=True, name='mandi-loader'
            )
            _mandi['thread'] = current
            current.start()
            running = True
        if running and not _mandi['rows']:
            thread = current
    if thread is not None and wait:
        thread.join(timeout=wait)

def query_mandi(commodity: str = '', state: str = '', district: str = '', sort: str = 'newest') -> list[dict]:
    rows = _mandi['rows']
    c, s, d = commodity.casefold(), state.casefold(), district.casefold()
    out = [
        row for row in rows
        if (not c or c in row['commodity'].casefold())
        and (not s or s in row['state'].casefold())
        and (not d or d in row['district'].casefold())
    ]
    if sort == 'high':
        out.sort(key=lambda row: (row['modal_price'] is None, -(row['modal_price'] or 0)))
    elif sort == 'low':
        out.sort(key=lambda row: (row['modal_price'] is None, row['modal_price'] or 0))
    elif sort == 'az':
        out.sort(key=lambda row: (row['commodity'].casefold(), row['state'].casefold(), row['market'].casefold()))
    else:
        out.sort(key=lambda row: (-row['_date'].toordinal(), row['commodity'].casefold(), row['state'].casefold(), row['market'].casefold()))
    return out

def mandi_stats(rows: list[dict]) -> dict | None:
    modal = [row['modal_price'] for row in rows if row['modal_price'] is not None]
    if not modal:
        return None
    return {
        'markets': len({(row['state'], row['market']) for row in rows}),
        'low': min(modal), 'high': max(modal), 'avg': sum(modal) / len(modal)
    }

ensure_mandi()
# documents
@lru_cache(maxsize=8)
def _pdf_text(path_str: str, mtime: float) -> str:
    path = Path(path_str)
    try:
        import fitz
        with fitz.open(path) as doc: return '\n'.join(p.get_text() for p in doc).strip()
    except ImportError:
        pass
    except Exception as exc:
        app.logger.warning('PyMuPDF failed for %s: %s', path.name, exc)
    try:
        from pypdf import PdfReader
        return '\n'.join((p.extract_text() or '') for p in PdfReader(str(path)).pages).strip()
    except Exception as exc:
        app.logger.warning('PDF extraction failed for %s: %s', path.name, exc)
        return ''

def extract_pdf_text(path: Path) -> str:
    return _pdf_text(str(path), path.stat().st_mtime)

# final webpages
@app.context_processor
def inject_globals():
    return {'now_year': datetime.now().year, 'languages': LANGUAGES}

@app.get('/')
def home():
    return render_template('index.html', active='home', ai_ready=AI_CONFIGURED, mandi_ready=True, model_ready=(BASE_DIR / 'model.onnx').exists())

@app.get('/tank')
@app.get('/rover')
def tank():
    """Rover telemetry page. Both legacy and new URLs are supported."""
    return render_template('tank.html', active='tank')

@app.get('/market')
def market():
    f = {k: request.args.get(k, '').strip()[:80] for k in ('commodity', 'state', 'district')}
    sort = request.args.get('sort', 'newest'); sort = sort if sort in MANDI_SORTS else 'newest'
    page = request.args.get('page', 1, type=int) or 1
    ensure_mandi(wait=0 if _mandi['rows'] else 8)
    with _mandi_lock:
        info = {k: _mandi[k] for k in ('error', 'partial', 'loaded_wall', 'latest', 'states', 'commodities', 'districts')}
        loading = bool(_mandi['thread'] and _mandi['thread'].is_alive())
    error = info['error'] if not _mandi['rows'] else None
    rows = query_mandi(sort=sort, **f)
    total = len(rows); pages = max(1, -(-total // MANDI_PER_PAGE)); page = min(max(1, page), pages)
    start = (page - 1) * MANDI_PER_PAGE; chunk = rows[start:start + MANDI_PER_PAGE]
    link = lambda p: url_for('market', page=p if p > 1 else None, sort=sort if sort != 'newest' else None, **{k: v for k, v in f.items() if v})
    state_key = next((s for s in info['districts'] if s.casefold() == f['state'].casefold()), None)
    return render_template('market.html', active='market', data=chunk, error=error, notice=info['error'] if _mandi['rows'] else None, loading=loading and not _mandi['rows'],
                           states=info['states'] or STATES, commodities=info['commodities'] or COMMODITIES, districts=info['districts'].get(state_key, []),
                           total=total, dataset_total=len(_mandi['rows']), page=page, pages=pages, first=start + 1 if chunk else 0, last=start + len(chunk),
                           prev_url=link(page - 1) if page > 1 else None, next_url=link(page + 1) if page < pages else None, stats=mandi_stats(rows),
                           sort=sort, sorts=MANDI_SORTS, latest=info['latest'], partial=info['partial'],
                           loaded_at=info['loaded_wall'].strftime('%d %b %Y, %I:%M %p') if info['loaded_wall'] else '', **f)

@app.route('/advisor', methods=['GET', 'POST'])
def advisor():
    src = request.form if request.method == 'POST' else request.args
    form = {k: src.get(k, '').strip()[:1000] for k in ('crop', 'location', 'stage', 'question')}
    form['language'] = src.get('language', 'English') if src.get('language') in LANGUAGES else 'English'
    answer = provider = None; ok = True
    if request.method == 'POST':
        prompt = (f'Crop: {form["crop"]}\nLocation: {form["location"] or "Not provided"}\nGrowth stage: {form["stage"] or "Not provided"}\n'
                  f'Farmer question: {form["question"] or "Give a practical crop-care plan."}\n'
                  'Provide a concise plan with immediate checks, irrigation/nutrition considerations, likely risks, and what additional information would improve the advice.')
        answer, provider, ok = ai_answer(prompt, 'advisor', form['language'])
    return render_template('advisor.html', active='advisor', answer=answer, provider=provider, ok=ok, form=form)

@app.route('/ai', methods=['GET', 'POST'])
def ai_scan():
    result = None; error = None
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or not file.filename:
            error = 'Choose a plant image first.'
        elif Path(secure_filename(file.filename)).suffix.lower().lstrip('.') not in ALLOWED_IMAGE_EXTENSIONS:
            error = 'Use a JPG, JPEG, PNG or WEBP image.'
        else:
            session = get_model()
            if session is None:
                error = _model_error or 'The leaf scanner is currently unavailable.'
            else:
                try:
                    from PIL import Image, ImageDraw, ImageOps
                    image = ImageOps.exif_transpose(Image.open(io.BytesIO(file.read()))).convert('RGB')
                    if image.width * image.height > 25_000_000:
                        raise ValueError('Image is too large. Please use a smaller photo.')
                    detections = _run_onnx_scan(session, image, conf_thresh=0.20)
                    if not detections:
                        result = {
                            'disease': 'No confident detection',
                            'confidence': None,
                            'note': 'No clear leaf condition was detected above the confidence threshold. Try a well-lit close-up of a single affected leaf.',
                            'detections': [],
                            'image': None,
                            'crop': 'Tomato',
                        }
                    else:
                        best = detections[0]
                        label = best['label']
                        seen: dict[str, float] = {}
                        for d in detections:
                            seen.setdefault(d['label'], round(d['confidence'] * 100, 1))
                        lower = label.lower()
                        note = next((v for k, v in DISEASE_NOTES.items() if k in lower), GENERIC_NOTE)

                        annotated = image.copy()
                        draw = ImageDraw.Draw(annotated)
                        stroke = max(3, int(min(annotated.size) * 0.006))
                        for d in detections[:8]:
                            x1, y1, x2, y2 = d['box']
                            draw.rectangle([x1, y1, x2, y2], outline=(41, 74, 53), width=stroke)
                            tag = f"{d['label']} {round(d['confidence'] * 100)}%"
                            draw.rectangle([x1, max(0, y1 - 20), min(annotated.width, x1 + len(tag) * 7 + 10), y1], fill=(41, 74, 53))
                            draw.text((x1 + 5, max(2, y1 - 16)), tag, fill=(255, 254, 250))

                        annotated.thumbnail((900, 900))
                        buf = io.BytesIO()
                        annotated.save(buf, 'JPEG', quality=85)
                        result = {
                            'disease': label,
                            'confidence': round(best['confidence'] * 100, 1),
                            'note': note,
                            'detections': [{'label': k, 'confidence': v} for k, v in list(seen.items())[:5]],
                            'image': 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode(),
                            'healthy': 'healthy' in lower,
                            'crop': 'Tomato',
                        }
                except Exception as exc:
                    app.logger.exception('Crop scan failed')
                    error = str(exc) if isinstance(exc, ValueError) else 'Could not analyze this image. Check that the file is a valid photo.'
    return render_template('ai.html', active='ai', result=result, error=error, model_available=(BASE_DIR / 'model.onnx').exists())

@app.get('/legal')
def legal():
    docs = []
    for filename, label, language in [('MSPE.pdf', 'MSP Guide', 'English'), ('MSPH.pdf', 'MSP Guide', 'Hindi')]:
        docs.append({'filename': filename, 'label': label, 'language': language, 'available': (BASE_DIR / 'static' / filename).exists()})
    return render_template('legal.html', active='legal', docs=docs)

# JSON APIs
@app.post('/api/legal/summarize')
@rate_limited(10)
def legal_summarize():
    data = request.get_json(silent=True) or {}
    language = (data.get('language') or 'English').strip()[:40]
    selected = data.get('document', 'MSPE.pdf')
    question = str(data.get('question', '')).strip()[:1000]
    if selected not in {'MSPE.pdf', 'MSPH.pdf'}: return jsonify(error='Unknown document.'), 400
    path = BASE_DIR / 'static' / selected
    if not path.exists(): return jsonify(error='The selected PDF is missing.'), 404
    content = extract_pdf_text(path)
    if not content: return jsonify(error='Could not extract readable text from this PDF. It may be scanned; OCR would be needed. (Install PyMuPDF or pypdf.)'), 422
    task = (f'Answer this farmer\'s question using ONLY the document text below. If the document does not answer it, say so.\n\nQUESTION: {question}\n\n' if question else
            'Summarize the following source document for a farmer. Include the main points, who it applies to, any steps/conditions mentioned, and what the document does not establish. Do not add facts not present in the text.\n\n')
    answer, provider, ok = ai_answer(task + 'DOCUMENT TEXT:\n' + content[:24000], 'legal', language)
    return jsonify(response=answer, provider=provider, source=selected, ok=ok), (200 if ok else 503)

@app.post('/api/chat')
@rate_limited(20)
def api_chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get('message', '')).strip()
    if not message: return jsonify(error='Message is required.'), 400
    if len(message) > 5000: return jsonify(error='Message is too long.'), 413
    context = data.get('context', 'general')
    if context not in SYSTEMS: context = 'general'
    language = str(data.get('language', 'English'))[:40]
    history = []
    for turn in (data.get('history') or [])[-8:]:
        if not (isinstance(turn, dict) and turn.get('role') in ('user', 'assistant') and isinstance(turn.get('content'), str)): continue
        item = {'role': turn['role'], 'content': turn['content'][:3000]}
        rd = turn.get('reasoning_details')
        if turn['role'] == 'assistant' and isinstance(rd, list) and len(json.dumps(rd)) < 30000: item['reasoning_details'] = rd
        history.append(item)
    r = ai_reply(message, context, language, history)
    return jsonify(response=r['text'], provider=r['provider'], ok=r['ok'], reasoning_details=r['reasoning_details']), (200 if r['ok'] else 503)

@app.get('/api/market')
def api_market():
    f = {k: request.args.get(k, '').strip()[:80] for k in ('commodity', 'state', 'district')}
    sort = request.args.get('sort', 'newest'); sort = sort if sort in MANDI_SORTS else 'newest'
    per = min(max(request.args.get('per_page', MANDI_PER_PAGE, type=int) or MANDI_PER_PAGE, 1), 200)
    page = max(request.args.get('page', 1, type=int) or 1, 1)
    ensure_mandi(wait=0 if _mandi['rows'] else 8)
    rows = query_mandi(sort=sort, **f)
    if not _mandi['rows']:
        return jsonify(records=[], total=0, error=_mandi['error'] or 'Mandi data is still loading.'), 503
    return jsonify(records=[_public(r) | {'arrival_date': r['arrival_date']} for r in rows[(page - 1) * per:page * per]], total=len(rows), page=page, per_page=per, error=None)

@app.get('/api/telemetry')
def telemetry_get():
    with _telemetry_lock: data = dict(_latest_telemetry)
    fresh = bool(data.get('received_at') and time.monotonic() - _last_telemetry_monotonic < 30)
    data['connected'] = fresh
    if not fresh: data['source'] = 'Waiting for rover telemetry'
    return jsonify(data)

@app.post('/api/telemetry')
def telemetry_post():
    global _last_telemetry_monotonic, _latest_telemetry
    if TELEMETRY_TOKEN and request.headers.get('X-Device-Token', '') != TELEMETRY_TOKEN:
        return jsonify(error='Unauthorized device.'), 401
    data = request.get_json(silent=True)
    if not isinstance(data, dict): return jsonify(error='Send a JSON object.'), 400
    allowed = {'battery_voltage', 'battery_percent', 'tank_a_percent', 'tank_b_percent', 'tank_c_percent', 'latitude', 'longitude', 'speed_kmh', 'area_ha', 'status', 'runtime_seconds'}
    clean = {k: data[k] for k in allowed if k in data}
    for k, v in clean.items():
        if k == 'status': clean[k] = str(v)[:60]
        elif not isinstance(v, (int, float)) or isinstance(v, bool): return jsonify(error=f'{k} must be numeric.'), 400
    for k in ('battery_percent', 'tank_a_percent', 'tank_b_percent', 'tank_c_percent'):
        if k in clean and not 0 <= clean[k] <= 100: return jsonify(error=f'{k} must be between 0 and 100.'), 400
    if 'latitude' in clean and not -90 <= clean['latitude'] <= 90: return jsonify(error='latitude out of range.'), 400
    if 'longitude' in clean and not -180 <= clean['longitude'] <= 180: return jsonify(error='longitude out of range.'), 400
    clean.update({'connected': True, 'received_at': datetime.now(timezone.utc).isoformat(), 'source': 'Rover device'})
    with _telemetry_lock:
        _latest_telemetry = clean
        _last_telemetry_monotonic = time.monotonic()
    return jsonify(ok=True, received_at=clean['received_at'])

@app.get('/health')
def health():
    return jsonify(status='ok', ai_configured=AI_CONFIGURED, openrouter=bool(OPENROUTER_API_KEY), ai_model=OPENROUTER_MODEL,
                   mandi_configured=True, mandi_api_base=MANDI_API_BASE, mandi_rows_cached=len(_mandi['rows']), mandi_rows=len(_mandi['rows']), mandi_error=_mandi['error'], model_file=(BASE_DIR / 'model.onnx').exists(), telemetry_token_set=bool(TELEMETRY_TOKEN))

#errors
@app.errorhandler(413)
def too_large(_): return render_template('error.html', active='', code=413, message='The uploaded file is too large. Maximum size is 10 MB.'), 413

@app.errorhandler(404)
def not_found(_): return render_template('error.html', active='', code=404, message='That page does not exist.'), 404

@app.errorhandler(500)
def server_error(_): return render_template('error.html', active='', code=500, message='Something went wrong on our side. Please try again.'), 500

if __name__ == '__main__':
    app.run(host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '5000')), debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true')
