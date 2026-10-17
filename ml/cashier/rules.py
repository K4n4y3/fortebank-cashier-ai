"""Правила T1–T12 поверх временной линии восприятия.

Каждое правило — явная проверка последовательности состояний/событий; таймкод берётся из кадра события.
Описания (observation) формулируются только через видимое, без выводов о суммах и документах.
"""
from dataclasses import dataclass, field

import numpy as np

from ml.cashier.perception import Timeline


@dataclass
class Finding:
    type_id: str
    t: float
    confidence: float
    observation: str
    evidence_t: list[float] = field(default_factory=list)
    rule: str = ""


@dataclass
class Params:
    fps: float = 2.0
    min_seg: int = 2            # минимальная длина сегмента состояния, кадров
    door_open_sec: float = 4.0  # сколько дверь должна быть открыта, чтобы это было нарушением
    lookback_sec: float = 3.0   # окно «до события» для проверки состояния
    merge_sec: float = 5.0      # одинаковые типы ближе этого сливаются
    use_other: bool = False     # добавлять ли свободные находки VLM (OTHER)


class _TL:
    """Удобный доступ к массивам временной линии."""

    def __init__(self, tl: Timeline):
        self.t = np.array(tl.times, dtype=float)
        n = len(self.t)

        def arr(key, fill=0.0):
            vals = tl.state[key]
            out, last = np.zeros(n), fill
            for i, v in enumerate(vals):
                if v is not None:
                    last = v
                out[i] = last  # пропуски заполняем последним известным значением
            return out

        self.in_room = _smooth(arr("cashier_in_room", 1.0) >= 0.5)
        self.at_work = _smooth(arr("cashier_at_workplace", 1.0) >= 0.5)
        self.visitors = np.round(_median3(arr("visitors", 0.0))).astype(int)
        self.safe_open = _smooth(arr("safe_open", 0.0) >= 0.5)
        self.door_open = _smooth(arr("door_open", 0.0) >= 0.5)
        self.cash_desk = _smooth(arr("cash_on_desk", 0.0) >= 0.5)
        self.items = _smooth(arr("personal_items_on_desk", 0.0) >= 0.5)
        hands = list(tl.state["cashier_hands"])
        last = "empty"
        for i, h in enumerate(hands):
            last = h if h is not None else last
            hands[i] = last
        self.hands = np.array(hands, dtype=object)
        self.events = tl.events
        self.other = tl.other

    def idx(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))

    def span(self, t0: float, t1: float) -> slice:
        return slice(self.idx(t0), self.idx(t1) + 1)

    def ev(self, name: str, t0: float = -1e9, t1: float = 1e9, actor: str | None = None) -> list[dict]:
        return [e for e in self.events if e["event"] == name and t0 <= e["t"] <= t1 and (actor is None or e["actor"] == actor)]


def _smooth(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x.astype(bool)
    y = x.astype(int)
    m = np.convolve(np.pad(y, 1, mode="edge"), np.ones(3, dtype=int), mode="valid")
    return m >= 2


def _median3(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return x
    p = np.pad(x, 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def _segments(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    segs, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                segs.append((start, i - 1))
            start = None
    return segs


def _sessions(x: _TL, p: Params) -> list[dict]:
    """Визиты: отрезки, где в комнате есть посетители. Курьерские визиты помечаются."""
    out = []
    for s, e in _segments(x.visitors >= 1, p.min_seg):
        t0, t1 = x.t[s], x.t[e]
        courier = bool(x.ev("cash_bag_given_to_cashier", t0 - 1, t1 + 1) or
                       [ev for ev in x.events if ev["actor"] == "courier" and t0 - 1 <= ev["t"] <= t1 + 1] or
                       (x.hands[s:e + 1] == "cash_bag").any())
        out.append(dict(s=s, e=e, t0=t0, t1=t1, courier=courier))
    return out


def apply(tl: Timeline, p: Params | None = None) -> list[Finding]:
    p = p or Params()
    x = _TL(tl)
    if len(x.t) == 0:
        return []
    lb = int(round(p.lookback_sec * p.fps))
    F: list[Finding] = []
    sessions = _sessions(x, p)

    # T1 — кассир покидает помещение, оставив сейф открытым или наличные на столе
    for k in range(1, len(x.t)):
        if x.in_room[k - 1] and not x.in_room[k] and not x.in_room[k:k + p.min_seg].any():
            win = slice(max(0, k - lb), min(len(x.t), k + 2))
            safe, cash = x.safe_open[win].any(), x.cash_desk[win].any()
            if safe or cash:
                # начало ухода: последний кадр «на рабочем месте» в пределах lookback
                j = next((i for i in range(k - 1, max(-1, k - 2 * lb), -1) if x.at_work[i]), k - 1)
                t = x.t[min(j + 1, k)]
                what = "сейф остаётся открытым" if safe else "на столе остаются наличные"
                if safe and cash:
                    what = "сейф остаётся открытым, на столе остаются наличные"
                F.append(Finding("T1", t, 0.75, f"Кассир покидает помещение; {what}", [t, x.t[k]], "leave_unattended"))

    # T2 — вход человека при открытых ценностях / два посетителя одновременно
    # (вход должен держаться >= 3 кадров: на кадре прохода через дверь модель часто «удваивает» людей)
    hold = p.min_seg + 1
    for k in range(1, len(x.t) - hold + 1):
        if x.visitors[k] > x.visitors[k - 1] and (x.visitors[k:k + hold] >= x.visitors[k]).all():
            t = x.t[k]
            prev = slice(max(0, k - lb), k)
            entering_courier = any(e["actor"] == "courier" for e in x.ev("person_enters_room", t - 2, t + 2))
            if entering_courier:
                continue
            if x.visitors[k] >= 2:
                F.append(Finding("T2", t, 0.7, "В кассовое помещение входит ещё один человек; в помещении одновременно "
                                                "находятся несколько посетителей", [t], "second_visitor"))
            elif x.cash_desk[prev].any() or (x.hands[prev] == "cash").any():
                F.append(Finding("T2", t, 0.65, "В кассовое помещение входит человек, когда на столе или в руках "
                                                 "кассира находятся наличные", [t], "enter_with_cash_open"))

    # T3 — дверь кассового помещения остаётся открытой
    for s, e in _segments(x.door_open & x.in_room, int(round(p.door_open_sec * p.fps))):
        t = x.t[s]
        F.append(Finding("T3", t, 0.6, f"Дверь кассового помещения остаётся открытой около {x.t[e] - x.t[s]:.0f} с, "
                                        "кассир находится внутри", [t, x.t[min(e, s + 4)]], "door_open"))

    # T4 / T5 — телефон, еда/напитки в руках кассира
    for hand, typ, text in (("phone", "T4", "Кассир держит в руках мобильный телефон на рабочем месте"),
                            ("cup_or_food", "T5", "Кассир пьёт или ест на рабочем месте")):
        for s, e in _segments(x.hands == hand, p.min_seg):
            F.append(Finding(typ, x.t[s], 0.7, text, [x.t[s], x.t[(s + e) // 2]], f"hands_{hand}"))

    # T6 — личные вещи на столе
    for s, e in _segments(x.items, p.min_seg + 1):
        F.append(Finding("T6", x.t[s], 0.65, "На рабочем столе кассира находятся личные вещи (сумка/одежда)",
                         [x.t[s], x.t[min(e, s + 4)]], "personal_items"))
    for ev in x.ev("personal_item_put_on_desk"):
        F.append(Finding("T6", ev["t"], 0.6, "Кассир кладёт личную вещь на рабочий стол", [ev["t"]], "personal_item_event"))

    # T7 — инкассатор уходит до вскрытия сумки
    for ev in x.ev("cash_bag_given_to_cashier"):
        t = ev["t"]
        left = [s for s in sessions if s["t0"] <= t + 1 and t <= s["t1"] <= t + 8]
        opened = x.ev("cash_bag_opened", t, (left[0]["t1"] if left else t + 8))
        if left and not opened:
            F.append(Finding("T7", t, 0.7, "Инкассатор передаёт сумку и покидает помещение до её вскрытия и проверки",
                             [t, left[0]["t1"]], "courier_left"))

    # T8 — купюра уходит в одежду кассира
    for ev in x.ev("banknote_put_into_clothes"):
        F.append(Finding("T8", ev["t"], 0.6, "Кассир перемещает купюру к одежде/рукаву, а не на стол или в хранилище",
                         [ev["t"]], "banknote_clothes"))

    # T9 / T10 / T12 — процедура обслуживания клиента в пределах визита
    for s in sessions:
        if s["courier"]:
            continue
        t0, t1 = s["t0"], s["t1"]
        cash_in = x.ev("cash_given_to_cashier", t0 - 1, t1 + 1)
        cash_out = x.ev("cash_given_to_visitor", t0 - 1, t1 + 1)
        first_cash = min([e["t"] for e in cash_in + cash_out], default=None)
        if first_cash is not None:
            id_seen = (x.ev("id_card_given_to_cashier", t0 - 1, first_cash) or x.ev("cashier_inspects_id_card", t0 - 1, first_cash)
                       or (x.hands[x.span(t0 - 1, first_cash)] == "id_card").any())
            if not id_seen:
                F.append(Finding("T9", first_cash, 0.6, "Наличные передаются без предшествующей видимой передачи и "
                                                         "осмотра удостоверения личности клиента", [first_cash], "no_id_check"))
        if cash_in and cash_out and max(e["t"] for e in cash_out) > min(e["t"] for e in cash_in) + 2:
            t = min(e["t"] for e in cash_out if e["t"] > min(c["t"] for c in cash_in) + 2)
            F.append(Finding("T12", t, 0.55, "После пересчёта кассир возвращает часть наличных клиенту; правомерность "
                                              "возврата по видео проверить нельзя", [t], "cash_returned"))
        end = s["e"]
        tail = slice(max(0, end - 2), min(len(x.t), end + 2))
        recent_out = x.ev("cash_given_to_visitor", t1 - 4, t1 + 1)
        if (x.hands[tail] == "cash").sum() >= 2 and not recent_out and cash_in:
            F.append(Finding("T10", t1, 0.55, "Клиент покидает кассу, пока кассир ещё работает с его наличными",
                             [t1], "client_left_early"))

    # T11 — наличные предыдущего клиента не убраны, новые кладутся сверху
    for a, b in zip(sessions, sessions[1:]):
        gap = slice(a["e"], b["s"] + 1)
        if x.cash_desk[gap].all():
            new_cash = x.ev("cash_given_to_cashier", b["t0"] - 1, b["t1"] + 1)
            if new_cash:
                t = new_cash[0]["t"]
                F.append(Finding("T11", t, 0.6, "Наличные предыдущего посетителя остаются на столе, наличные следующего "
                                                 "клиента принимаются поверх них", [t, b["t0"]], "cash_mixing"))

    if p.use_other:
        for o in x.other:
            F.append(Finding("OTHER", o["t"], 0.5, o["text"], [o["t"]], "vlm_other"))

    return _merge(F, p.merge_sec)


def _merge(found: list[Finding], gap: float) -> list[Finding]:
    out: list[Finding] = []
    for f in sorted(found, key=lambda f: (f.type_id, f.t)):
        if out and out[-1].type_id == f.type_id and f.t - out[-1].t <= gap:
            out[-1].confidence = max(out[-1].confidence, f.confidence)
            continue
        out.append(f)
    return sorted(out, key=lambda f: f.t)
