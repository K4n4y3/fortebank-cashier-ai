"""Первая ступень: поиск кандидатов-нарушений VLM по перекрывающимся окнам кадров.

Ролик режется на окна (по умолчанию 10 с, шаг 5 с, 2 кадра/с). VLM получает кадры с подписями
времени и чек-лист типов, отвечает строгим JSON. Ответы окон сливаются по типу и времени.
"""
import concurrent.futures as cf
import json
import re
import urllib.request
from pathlib import Path

import numpy as np

from ml.cashier.video import Frame, sample

TYPES = {
    "T1": "The cashier leaves the workplace/room while the safe is open or cash, a cash bag or valuables stay unattended on the desk.",
    "T2": "A person enters the cash room while cash is lying open on the desk or the cashier is handling cash at the safe/counting machine "
          "for someone else (e.g. the next client walks in before the previous client's cash is put away), or two visitors are in the room at once.",
    "T3": "The door of the cash room is left open while the cashier works with cash.",
    "T4": "The cashier holds, uses or looks at a smartphone (a small dark rectangle in her hands) at the workplace.",
    "T5": "The cashier eats or drinks at the cash desk.",
    "T6": "Personal belongings (handbag, coat, clothes) are placed on the cash desk.",
    "T7": "A cash-in-transit courier hands over a cash bag and leaves before the cashier opens and checks it in his presence.",
    "T8": "The cashier moves a banknote towards her own clothes, sleeve, pocket or body instead of the desk, counter or safe.",
    "T9": "Cash is accepted from or paid to a client without the client first handing over an identity card that the cashier visibly inspects.",
    "T10": "The client walks away before the cashier has finished counting/processing the operation.",
    "T11": "Cash from the previous client is not put away and the next client's cash is placed on top of it (mixing cash of different clients).",
    "T12": "After counting, the cashier hands part of the cash back to the client.",
    "T13": "After locking the safe the cashier walks away with empty hands and leaves the key in the safe lock "
           "(instead of taking the keys with her).",
    "OTHER": "Any other clearly visible breach of cash handling or cash-room security rules.",
}

SYSTEM = """You review CCTV footage from a bank cash room (top-down view, one camera).
People: the cashier (sits behind the desk), clients, cash-in-transit couriers.
Objects: a grey metal safe, a banknote counting machine, a laptop, the room's glass door.
These situations are NORMAL and must NOT be reported:
- one client walks into the room, sits down at the desk, is served and leaves;
- the client hands over an ID card, the cashier looks at it, then cash is handed over and counted (by hand or machine);
- the cashier opens the safe, puts cash in or takes it out, and closes the safe;
- the cashier gets up and leaves the room after closing the safe, taking the keys, with no cash left on the desk;
- a cash-in-transit courier brings a bag and stays while the cashier opens and checks it;
- the cashier types on the laptop or counts banknotes at the desk.

Report ONLY violations that are clearly visible in the frames. Do not guess about amounts,
documents or anything not visible. If nothing is wrong, return an empty list. Use confidence
above 0.7 only when the violation is unambiguous.

Violation types:
{types}

Answer with JSON only:
{{"events": [{{"type": "T1".."T13" or "OTHER", "frame": <index of the frame where it starts>,
"confidence": <0..1>, "observation": "<one short sentence IN RUSSIAN: what exactly is visible>"}}]}}"""


def _ask(api: str, model: str, frames: list[Frame], timeout: int = 300) -> tuple[list[dict], str]:
    content = [{"type": "text", "text": f"Frames below, numbered from 0; the clip starts at t={frames[0].t:.1f}s."}]
    for k, f in enumerate(frames):
        content.append({"type": "text", "text": f"[frame {k}, t={f.t:.1f}s]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f.jpeg_b64}"}})
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM.format(types="\n".join(f"{k}: {v}" for k, v in TYPES.items()))},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(f"{api}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001 — одно упавшее окно не должно ронять весь ролик
        return [], f"ERROR {e.__class__.__name__}: {e}"
    m = re.search(r"\{.*\}", text, re.S)
    try:
        events = json.loads(m.group(0)).get("events", []) if m else []
    except json.JSONDecodeError:
        events = []
    out = []
    for e in events:
        try:
            typ = str(e.get("type", "OTHER"))
            k = min(max(int(e.get("frame", 0)), 0), len(frames) - 1)
            out.append(dict(type=typ if typ in TYPES else "OTHER", t=frames[k].t,
                            conf=float(e.get("confidence", 0.5)), obs=str(e.get("observation", ""))))
        except (TypeError, ValueError):
            continue
    return out, text


def merge(events: list[dict], gap: float = 4.0) -> list[dict]:
    """Одинаковые типы ближе gap секунд — одно событие (берём самое раннее время, лучшую уверенность)."""
    events = sorted(events, key=lambda e: (e["type"], e["t"]))
    merged: list[dict] = []
    for e in events:
        if merged and merged[-1]["type"] == e["type"] and e["t"] - merged[-1]["t"] <= gap:
            if e["conf"] > merged[-1]["conf"]:
                merged[-1].update(conf=e["conf"], obs=e["obs"])
            continue
        merged.append(dict(e))
    return sorted(merged, key=lambda e: e["t"])


def detect(path: Path, api: str, model: str, fps: float = 2.0, window: float = 10.0, stride: float = 5.0,
           long_side: int = 848, parallel: int = 16, thr: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Кандидаты ролика: [{type, t, conf, obs}] с conf >= thr, и сырые ответы окон (для отладки)."""
    frames, _ = sample(path, fps, long_side)
    if not frames:
        return [], []
    dur = frames[-1].t
    starts = list(np.arange(0, max(dur - window, 0) + 1e-6, stride)) or [0.0]
    if starts[-1] + window < dur:  # последнее окно должно дойти до конца ролика
        starts.append(max(0.0, dur - window))
    wins = [[f for f in frames if s <= f.t < s + window] for s in starts]
    wins = [w for w in wins if w]
    found, raw = [], []
    with cf.ThreadPoolExecutor(parallel) as ex:
        for (evs, text), w in zip(ex.map(lambda w: _ask(api, model, w), wins), wins):
            found += evs
            raw.append({"start": w[0].t, "answer": text})
    failed = [r for r in raw if r["answer"].startswith("ERROR")]
    if len(failed) == len(raw):
        # модель не ответила ни на одно окно — нельзя молча вернуть «нарушений нет»
        raise RuntimeError(f"VLM недоступен ({api}): {failed[0]['answer'][:200]}")
    return [e for e in merge(found) if e["conf"] >= thr], raw
