"""Вторая ступень: проверка кандидата-нарушения узким вопросом по его типу.

Для кандидата (тип, время) берётся клип вокруг таймкода, VLM находит на кадре кассира/стол/сейф/дверь,
нужная область вырезается и увеличивается. Затем задаётся вопрос, сформулированный как утверждение
о нарушении; ответ "yes" подтверждает кандидата. Неуверенность ("unclear") не подтверждает.
"""
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ml.cashier.video import encode, grab, resize

# какие кадры брать вокруг кандидата: (секунд до, секунд после, fps)
CLIP = {
    "T1": (4, 6, 2), "T2": (4, 4, 2), "T3": (3, 6, 2), "T4": (2, 4, 2), "T5": (2, 4, 2), "T6": (2, 4, 2),
    "T7": (3, 8, 2), "T8": (3, 3, 2), "T9": (16, 3, 1), "T10": (6, 4, 1.5), "T11": (10, 4, 1), "T12": (8, 4, 1.5),
    "OTHER": (3, 3, 2),
}
# какую область увеличивать
FOCUS = {"T4": "hands", "T5": "hands", "T8": "hands", "T9": "hands", "T12": "hands", "T10": "hands",
         "T11": "desk", "T6": "desk", "T1": "safe", "T3": "door"}

QUESTIONS = {
    "T1": "Is it true that the cashier walks out of the room (or away from the workplace out of the room) while the "
          "safe is left OPEN or banknotes / a cash bag are left lying on the desk? Say yes only if BOTH the leaving "
          "and the open safe / unattended cash are visible. Leaving after closing the safe with an empty desk is normal.",
    "T2": "Is it true that a person enters the cash room while banknotes lie open on the desk or the cashier handles "
          "cash for someone else, or that two visitors (besides the cashier) are inside at the same time? "
          "A single client entering a room with an empty desk to be served is normal.",
    "T3": "Is it true that the room's door stays open for several seconds while the cashier is inside working? "
          "Briefly opening the door to walk in or out is normal.",
    "T4": "Is it true that the cashier is holding or using a smartphone? Say yes only if a phone is clearly visible "
          "in her hands - not an ID card, not banknotes, not papers, not the counting machine.",
    "T5": "Is it true that the cashier is drinking or eating at the desk (a cup, bottle or food in her hand or at her mouth)?",
    "T6": "Is it true that a personal handbag, coat or clothing is lying on the cashier's desk?",
    "T7": "Is it true that a cash-in-transit courier hands over a cash bag and leaves the room BEFORE the cashier "
          "opens and checks the bag?",
    "T8": "Is it true that the cashier puts a banknote into her clothes, sleeve or pocket, or hides it at her body, "
          "instead of putting it on the desk, into the counting machine or the safe?",
    "T9": "These frames show a client being served. Is it true that cash is handed over (from the client to the "
          "cashier or back) and that BEFORE it no identity card was handed to the cashier and looked at? "
          "Say no only if you see an ID card handed over and inspected before the cash. An ID card is ONE small stiff "
          "WHITE/light plastic card the size of a bank card. Kazakh tenge BANKNOTES are paper, blue/green/pink/brown "
          "and larger - a blue banknote is NOT an ID card.",
    "T10": "Is it true that the client walks out of the room while the cashier is still counting or processing "
           "the client's cash?",
    "T11": "Is it true that banknotes from a previous operation are still lying on the desk when the next client's "
           "cash is placed on top of / next to them, so cash of different clients is mixed?",
    "T12": "Is it true that after counting the client's cash the cashier hands part of the banknotes back to the client?",
    "OTHER": "Is there a clearly visible breach of cash-handling or cash-room security rules in these frames?",
}

GROUND_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                       "minItems": 4, "maxItems": 4} for k in ("cashier", "desk", "safe", "door")},
    "required": ["cashier", "desk", "safe", "door"],
}
GROUND_PROMPT = ("Locate in this top-down CCTV image of a bank cash room: the cashier (the bank employee, include her "
                 "hands), the desk surface, the grey metal safe, and the room's glass door. Give each as a box "
                 "[x1, y1, x2, y2] in coordinates normalized to 0..1000 of the image width and height. "
                 "Use [0, 0, 0, 0] for anything not visible. Answer with JSON only.")

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"enum": ["yes", "no", "unclear"]},
        "frame": {"type": "integer"},
        "evidence": {"type": "string", "maxLength": 240},
    },
    "required": ["answer", "frame", "evidence"],
}


YNU = {"enum": ["yes", "no", "unclear"]}

# Типы, где один сложный вопрос работает плохо: модель отвечает на простые подвопросы, решение — в коде.
STRUCTURED = {
    "T1": {
        "clip": (5, 5, 1.5), "focus": None,
        "prompt": "Answer each field separately, based only on what is visible:\n"
                  "- cashier_leaves: does the cashier walk out of the room (through the door / out of view of the room)?\n"
                  "- leave_frame: index of the frame where she is leaving\n"
                  "- safe_open_when_leaving: at that moment, is the grey metal safe's door open?\n"
                  "- cash_on_desk_when_leaving: at that moment, do banknotes, cash bundles or a cash bag lie on the desk?",
        "schema": {"cashier_leaves": YNU, "leave_frame": {"type": "integer"},
                   "safe_open_when_leaving": YNU, "cash_on_desk_when_leaving": YNU},
        "frame_key": "leave_frame",
    },
    "T9": {
        "clip": (20, 3, 1), "focus": "hands",
        "prompt": "These frames show a client being served at the cash desk. Answer each field separately:\n"
                  "- cash_handover: are banknotes passed between the client and the cashier?\n"
                  "- cash_frame: index of the first frame where banknotes are passed\n"
                  "- id_card_before_cash: BEFORE that frame, does the client hand an identity card to the cashier and does "
                  "the cashier look at it? An ID card is ONE small stiff WHITE/light plastic card the size of a bank card. "
                  "Kazakh tenge banknotes are paper, blue/green/pink/brown and larger; a banknote is NOT an ID card.",
        "schema": {"cash_handover": YNU, "cash_frame": {"type": "integer"}, "id_card_before_cash": YNU},
        "frame_key": "cash_frame",
    },
}


def _decide(type_id: str, ans: dict, unclear_as_violation: bool) -> str:
    ok = lambda v: v == "yes" or (unclear_as_violation and v == "unclear")  # noqa: E731
    if type_id == "T1":
        if ans.get("cashier_leaves") != "yes":
            return "no"
        return "yes" if ok(ans.get("safe_open_when_leaving")) or ok(ans.get("cash_on_desk_when_leaving")) else "no"
    if type_id == "T9":
        if ans.get("cash_handover") != "yes":
            return "no"
        idb = ans.get("id_card_before_cash")
        return "yes" if idb == "no" or (unclear_as_violation and idb == "unclear") else "no"
    return ans.get("answer", "unclear")


@dataclass
class Verdict:
    confirmed: bool
    answer: str
    t: float
    evidence: str
    focus: str
    fields: dict | None = None  # сырые подответы структурированной проверки (для пересчёта политики без GPU)


def _post(api: str, body: dict, timeout: int = 180) -> dict | None:
    req = urllib.request.Request(f"{api}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(json.loads(r.read())["choices"][0]["message"]["content"])
    except Exception:  # noqa: BLE001
        return None


def ground(api: str, model: str, frame: np.ndarray) -> dict[str, tuple[float, float, float, float]]:
    body = {
        "model": model, "temperature": 0.0, "max_tokens": 120,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": GROUND_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode(resize(frame, 848))}"}},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "ground", "schema": GROUND_SCHEMA}},
    }
    ans = _post(api, body) or {}
    boxes = {}
    for k, b in ans.items():
        if isinstance(b, list) and len(b) == 4 and b[2] > b[0] and b[3] > b[1]:
            boxes[k] = tuple(v / 1000.0 for v in b)
    return boxes


def _crop(frame: np.ndarray, box, pad=0.15, out_long=896) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0.0, x1 - pad * bw), min(1.0, x2 + pad * bw)
    y1, y2 = max(0.0, y1 - pad * bh), min(1.0, y2 + pad * bh)
    c = frame[int(y1 * h):max(int(y1 * h) + 8, int(y2 * h)), int(x1 * w):max(int(x1 * w) + 8, int(x2 * w))]
    s = out_long / max(c.shape[:2])
    c = cv2.resize(c, (max(32, int(c.shape[1] * s)), max(32, int(c.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    return c


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


REF_NOTE = (" Before the frames to check you get short reference clips from earlier, already labeled footage of a "
            "cash room: a confirmed VIOLATION of this type and a NORMAL situation that looks similar. Compare the frames "
            "to check with them: answer yes only if they show the same kind of action as the violation reference; if "
            "they look like the normal reference, answer no. The references are only for comparison; answer about the "
            "frames to check, and frame indices refer to those frames.")


def verify(video: Path, type_id: str, t: float, duration: float, api: str, model: str,
           max_frames: int = 18, unclear_as_violation: bool = False,
           references: list[tuple[str, list[np.ndarray]]] | None = None, temperature: float = 0.0) -> Verdict:
    """references (v2) — размеченные примеры из памяти: [("violation" | "normal", кадры)], показываются модели
    перед проверяемыми кадрами, чтобы она сравнивала с уже виденными случаями, а не толковала правило сама."""
    st = STRUCTURED.get(type_id)
    if references:
        max_frames = min(max_frames, 14)  # чтобы вместе с примерами уложиться в лимит изображений на запрос
    before, after, fps = st["clip"] if st else CLIP.get(type_id, CLIP["OTHER"])
    t0, t1 = max(0.0, t - before), min(duration - 0.05, t + after)
    n = min(max_frames, max(2, int((t1 - t0) * fps) + 1))
    times = list(np.linspace(t0, t1, n))
    frames = grab(video, times, long_side=None)
    if not frames:
        return Verdict(False, "unclear", t, "no frames", "none")

    focus = st["focus"] if st else FOCUS.get(type_id)
    boxes = ground(api, model, frames[len(frames) // 2][1]) if focus else {}
    region = None
    if focus == "hands" and "cashier" in boxes:
        region = _union(boxes["cashier"], boxes["desk"]) if "desk" in boxes else boxes["cashier"]
    elif focus in ("desk", "safe", "door") and focus in boxes:
        region = boxes[focus]

    head = ("You verify a possible violation in bank cash-room CCTV. Frames are in time order. "
            + ("Each frame is a ZOOMED crop of the relevant area. " if region else ""))
    if st:
        text = head + st["prompt"] + "\nAlso give one-sentence evidence."
        schema = {"type": "object", "properties": {**st["schema"], "evidence": {"type": "string", "maxLength": 240}},
                  "required": list(st["schema"]) + ["evidence"]}
    else:
        text = (head + "Answer strictly based on what is visible.\nQuestion: " + QUESTIONS.get(type_id, QUESTIONS["OTHER"])
                + "\nReturn answer (yes/no/unclear), the index of the frame that shows it best, and one-sentence evidence.")
        schema = ANSWER_SCHEMA
    content = [{"type": "text", "text": text + (REF_NOTE if references else "")}]
    for label, ref_frames in references or []:
        content.append({"type": "text", "text": f"[Reference: {label.upper()} example]"})
        content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode(resize(f, 640))}"}}
                    for f in ref_frames]
    if references:
        content.append({"type": "text", "text": "[Frames to check]"})
    for k, (tt, fr) in enumerate(frames):
        img = _crop(fr, region) if region else resize(fr, 848)
        content.append({"type": "text", "text": f"[frame {k}, t={tt:.1f}s]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode(img)}"}})
        if region and type_id in ("T1", "T3") and k % 3 == 0:
            # для ухода/двери даём ещё и общий план — нужен контекст, где люди
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode(resize(fr, 640))}"}})
    body = {
        "model": model, "temperature": temperature, "max_tokens": 200,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "schema": schema}},
    }
    ans = _post(api, body, timeout=240) or {}
    a = _decide(type_id, ans, unclear_as_violation)
    k = ans.get(st["frame_key"] if st else "frame", len(frames) // 2)
    k = min(max(int(k) if isinstance(k, int) else 0, 0), len(frames) - 1)
    evidence = ans.get("evidence", "")
    if st:
        evidence = json.dumps({f: ans.get(f) for f in st["schema"]}, ensure_ascii=False) + " " + evidence
    return Verdict(a == "yes", a, frames[k][0], evidence, focus if region else "full",
                   {f: ans.get(f) for f in st["schema"]} if st else None)
