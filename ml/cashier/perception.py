"""VLM как «глаза»: по окнам кадров возвращает покадровое состояние сцены и атомарные события.

Нарушения здесь НЕ определяются — это делает rules.py по временной линии всего ролика.
VLM работает через локальный vLLM (OpenAI-совместимый API на 127.0.0.1), ответ ограничен JSON-схемой.
"""
import concurrent.futures as cf
import json
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

from ml.cashier.video import Frame

HANDS = ["empty", "cash", "phone", "id_card", "cup_or_food", "cash_bag", "papers", "other"]
TRI = ["yes", "no", "unknown"]
EVENTS = [
    "person_enters_room",
    "person_leaves_room",
    "id_card_given_to_cashier",
    "cashier_inspects_id_card",
    "cash_given_to_cashier",
    "cash_given_to_visitor",
    "cash_put_into_safe",
    "cash_taken_from_safe",
    "cash_bag_given_to_cashier",
    "cash_bag_opened",
    "banknote_put_into_clothes",
    "personal_item_put_on_desk",
]
ACTORS = ["cashier", "client", "courier", "other"]

SCHEMA = {
    "type": "object",
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "cashier_in_room": {"type": "boolean"},
                    "cashier_at_workplace": {"type": "boolean"},
                    "visitors": {"type": "integer", "minimum": 0, "maximum": 6},
                    "safe_open": {"enum": TRI},
                    "door_open": {"enum": TRI},
                    "cash_on_desk": {"type": "boolean"},
                    "cashier_hands": {"enum": HANDS},
                    "personal_items_on_desk": {"type": "boolean"},
                },
                "required": ["i", "cashier_in_room", "cashier_at_workplace", "visitors", "safe_open",
                             "door_open", "cash_on_desk", "cashier_hands", "personal_items_on_desk"],
            },
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "event": {"enum": EVENTS},
                    "actor": {"enum": ACTORS},
                },
                "required": ["i", "event", "actor"],
            },
        },
        "other_violation": {
            "type": "object",
            "properties": {"i": {"type": "integer"}, "text": {"type": "string", "maxLength": 200}},
            "required": ["i", "text"],
        },
    },
    "required": ["frames", "events", "other_violation"],
}

PROMPT = """You are the perception module of a bank cash-room CCTV analytics system.
The camera looks down at a small cash room: a desk with a banknote counting machine and a laptop,
a grey metal safe, a glass door. People: the CASHIER (bank employee who works behind the desk and
uses the safe) and VISITORS (clients, cash-in-transit couriers with a green cash bag, anyone else).

You get {n} frames numbered 0..{last} (t = seconds from video start). For EVERY frame report its state:
- cashier_in_room: the cashier is visible inside the room
- cashier_at_workplace: the cashier is at the desk or at the safe
- visitors: number of people inside the room other than the cashier (people behind the glass outside do not count)
- safe_open: is the safe door open ("unknown" if the safe is not visible)
- door_open: is the room's door open ("unknown" if not visible)
- cash_on_desk: banknotes or cash bundles are lying on the desk (not in hands, not in the machine)
- cashier_hands: what the cashier is holding: empty, cash, phone, id_card, cup_or_food, cash_bag, papers, other
- personal_items_on_desk: a handbag, coat or clothes lies on the desk

Then list atomic EVENTS with the frame index where each happens:
{events}

other_violation: if you clearly see something else that breaks cash-handling or cash-room security rules,
describe it in one short sentence with its frame index; otherwise i = -1 and text = "".

Describe only what is visible. Be precise about small objects in hands (phone vs ID card vs banknotes).
Answer with JSON only."""


@dataclass
class Timeline:
    times: list[float]
    state: dict[str, list]  # поле -> значения по кадрам (после слияния окон)
    events: list[dict]  # {t, event, actor}
    other: list[dict] = field(default_factory=list)  # {t, text}
    windows: int = 0
    errors: int = 0

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "Timeline":
        return Timeline(**d)


def _ask(api: str, model: str, frames: list[Frame], max_tokens: int) -> dict | None:
    content = [{"type": "text", "text": PROMPT.format(n=len(frames), last=len(frames) - 1,
                                                       events="\n".join(f"- {e}" for e in EVENTS))}]
    for k, f in enumerate(frames):
        content.append({"type": "text", "text": f"[frame {k}, t={f.t:.1f}s]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f.jpeg_b64}"}})
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"name": "perception", "schema": SCHEMA}},
    }
    req = urllib.request.Request(f"{api}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
        return json.loads(text)
    except Exception:  # noqa: BLE001 — окно без ответа просто пропускаем, слияние закроет дыру соседним окном
        return None


def perceive(frames: list[Frame], api: str, model: str, window: float = 10.0, stride: float = 5.0,
             parallel: int = 16, max_tokens: int = 2500) -> Timeline:
    times = [f.t for f in frames]
    dur = times[-1] if times else 0.0
    starts = list(np.arange(0, max(dur - window, 0) + 1e-6, stride)) or [0.0]
    if starts[-1] + window < dur:
        starts.append(max(0.0, dur - window))
    wins = [[k for k, f in enumerate(frames) if s <= f.t < s + window] for s in starts]
    wins = [w for w in wins if w]

    with cf.ThreadPoolExecutor(parallel) as ex:
        answers = list(ex.map(lambda w: _ask(api, model, [frames[k] for k in w], max_tokens), wins))

    votes: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    raw_events, other, errors = [], [], 0
    for w, ans in zip(wins, answers):
        if not ans:
            errors += 1
            continue
        for fr in ans.get("frames", []):
            i = fr.get("i")
            if not isinstance(i, int) or not 0 <= i < len(w):
                continue
            g = w[i]
            for key, val in fr.items():
                if key != "i":
                    votes[g][key].append(val)
        for ev in ans.get("events", []):
            i = ev.get("i")
            if isinstance(i, int) and 0 <= i < len(w):
                raw_events.append({"t": times[w[i]], "event": ev["event"], "actor": ev.get("actor", "other")})
        ov = ans.get("other_violation") or {}
        if isinstance(ov.get("i"), int) and 0 <= ov["i"] < len(w) and ov.get("text", "").strip():
            other.append({"t": times[w[ov["i"]]], "text": ov["text"].strip()})

    state = {k: [] for k in ["cashier_in_room", "cashier_at_workplace", "visitors", "safe_open", "door_open",
                             "cash_on_desk", "cashier_hands", "personal_items_on_desk"]}
    for g in range(len(frames)):
        v = votes.get(g, {})
        for key in state:
            vals = v.get(key, [])
            state[key].append(_combine(key, vals))
    return Timeline(times=times, state=state, events=_dedupe(raw_events), other=other,
                    windows=len(wins), errors=errors)


HANDOVERS = ["none", "id_card_to_cashier", "cash_to_cashier", "cash_to_visitor", "cash_bag_to_cashier"]

FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "cashier_in_room": {"type": "boolean"},
        "cashier_at_workplace": {"type": "boolean"},
        "visitors": {"type": "integer", "minimum": 0, "maximum": 6},
        "visitor_is_courier": {"type": "boolean"},
        "safe_open": {"type": "boolean"},
        "door_open": {"type": "boolean"},
        "cash_on_desk": {"type": "boolean"},
        "cashier_hands": {"enum": HANDS},
        "personal_items_on_desk": {"type": "boolean"},
        "handover": {"enum": HANDOVERS},
    },
    "required": ["cashier_in_room", "cashier_at_workplace", "visitors", "visitor_is_courier", "safe_open",
                 "door_open", "cash_on_desk", "cashier_hands", "personal_items_on_desk", "handover"],
}

FRAME_PROMPT = """You are the perception module of a bank cash-room CCTV system. Look at ONE camera frame
(top-down view of a small cash room: desk with a banknote counting machine and a laptop, a grey metal safe,
a glass door). The CASHIER is the bank employee working behind the desk / at the safe. VISITORS are everyone
else inside the room (clients; cash-in-transit couriers carry a green cash bag). People seen through the glass
walls outside the room are NOT inside.

Report for this frame:
- cashier_in_room: the cashier is inside the room (false if the room is empty or she is outside / at the door leaving)
- cashier_at_workplace: the cashier sits at the desk or stands at the safe
- visitors: number of visitors inside the room
- visitor_is_courier: a visitor carries a cash-in-transit bag
- safe_open: the SAFE is the grey metal cabinet (about knee-to-waist high box, often standing on a white
  stand next to the desk). It is open when its front door is swung out and you see the dark/inner side of the
  door or the inside of the box. If closed, it looks like a plain grey box.
- door_open: the room's glass door (with a metal handle) stands open / ajar rather than closed flush with the glass wall
- cash_on_desk: banknotes/bundles lie on the desk
- cashier_hands: what the cashier holds: empty, cash, phone, id_card, cup_or_food, cash_bag, papers, other
- personal_items_on_desk: a handbag, coat or clothes lies on the desk
- handover: an item is being passed between people right now: none, id_card_to_cashier, cash_to_cashier,
  cash_to_visitor, cash_bag_to_cashier
Look carefully at small objects in hands:
- phone: a smartphone held in one or both hands, the person looks down at its screen;
- id_card: ONE small stiff plastic card (white/light, size of a bank card), usually held up and looked at;
- cash: paper banknotes, often a stack or several notes; Kazakh tenge notes are blue, green, pink or brown
  and are larger and thinner than an ID card.
Answer with JSON only."""


def _ask_frame(api: str, model: str, frame: Frame) -> dict | None:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": FRAME_PROMPT},  # общий префикс первым — его кэширует vLLM
            {"type": "text", "text": f"Frame at t={frame.t:.1f}s:"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame.jpeg_b64}"}},
        ]}],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_schema", "json_schema": {"name": "frame", "schema": FRAME_SCHEMA}},
    }
    req = urllib.request.Request(f"{api}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(json.loads(r.read())["choices"][0]["message"]["content"])
    except Exception:  # noqa: BLE001
        return None


HANDOVER_EVENT = {"id_card_to_cashier": ("id_card_given_to_cashier", "client"),
                  "cash_to_cashier": ("cash_given_to_cashier", "client"),
                  "cash_to_visitor": ("cash_given_to_visitor", "cashier"),
                  "cash_bag_to_cashier": ("cash_bag_given_to_cashier", "courier")}


def perceive_frames(frames: list[Frame], api: str, model: str, parallel: int = 64) -> Timeline:
    """Покадровое восприятие: один кадр — один короткий запрос. События передач — из поля handover."""
    with cf.ThreadPoolExecutor(parallel) as ex:
        answers = list(ex.map(lambda f: _ask_frame(api, model, f), frames))
    keys = ["cashier_in_room", "cashier_at_workplace", "visitors", "safe_open", "door_open",
            "cash_on_desk", "cashier_hands", "personal_items_on_desk"]
    state = {k: [] for k in keys}
    courier, events, errors = [], [], 0
    prev_ho = "none"
    for f, ans in zip(frames, answers):
        if not ans:
            errors += 1
            for k in keys:
                state[k].append(None)
            courier.append(False)
            prev_ho = "none"
            continue
        for k in keys:
            state[k].append(_combine(k, [ans.get(k)]) if ans.get(k) is not None else None)
        courier.append(bool(ans.get("visitor_is_courier")))
        ho = ans.get("handover", "none")
        if ho != "none" and ho != prev_ho and ho in HANDOVER_EVENT:
            ev, actor = HANDOVER_EVENT[ho]
            events.append({"t": f.t, "event": ev, "actor": actor})
        prev_ho = ho
    # вход/выход курьера помечаем через события, чтобы правила отличали инкассатора от клиента
    for k in range(1, len(frames)):
        if courier[k] and not courier[k - 1]:
            events.append({"t": frames[k].t, "event": "person_enters_room", "actor": "courier"})
    return Timeline(times=[f.t for f in frames], state=state, events=sorted(events, key=lambda e: e["t"]),
                    other=[], windows=len(frames), errors=errors)


def _combine(key: str, vals: list):
    """Слияние ответов перекрывающихся окон для одного кадра."""
    if not vals:
        return None
    if key in ("safe_open", "door_open"):
        known = [v for v in vals if v in ("yes", "no") or isinstance(v, bool)]
        if not known:
            return None
        return float(np.mean([v is True or v == "yes" for v in known]))
    if key == "visitors":
        return float(np.median([int(v) for v in vals]))
    if key == "cashier_hands":
        c = Counter(vals)
        top = c.most_common()
        # при равенстве голосов предпочитаем непустое содержимое рук (полнота важнее)
        best = max(top, key=lambda kv: (kv[1], kv[0] != "empty"))
        return best[0]
    return float(np.mean([bool(v) for v in vals]))


def _dedupe(events: list[dict], gap: float = 2.0) -> list[dict]:
    out = []
    for e in sorted(events, key=lambda e: (e["event"], e["actor"], e["t"])):
        if out and out[-1]["event"] == e["event"] and out[-1]["actor"] == e["actor"] and e["t"] - out[-1]["t"] <= gap:
            continue
        out.append(e)
    return sorted(out, key=lambda e: e["t"])
