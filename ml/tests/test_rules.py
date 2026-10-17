"""Синтетические проверки правил: python -m ml.tests.test_rules"""
from ml.cashier import rules
from ml.cashier.perception import Timeline


def make(n, **over):
    base = dict(cashier_in_room=[1.0] * n, cashier_at_workplace=[1.0] * n, visitors=[0.0] * n,
                safe_open=[0.0] * n, door_open=[0.0] * n, cash_on_desk=[0.0] * n,
                cashier_hands=["empty"] * n, personal_items_on_desk=[0.0] * n)
    for k, v in over.items():
        base[k] = v
    return base


def tl(state, events=(), n=None):
    n = n or len(state["cashier_in_room"])
    return Timeline(times=[i * 0.5 for i in range(n)], state=state, events=list(events))


def types(found):
    return [(f.type_id, f.t) for f in found]


def test_leave_with_open_safe():
    n = 30
    s = make(n, cashier_in_room=[1.0] * 15 + [0.0] * 15, cashier_at_workplace=[1.0] * 12 + [0.0] * 18,
             safe_open=[1.0] * 16 + [0.0] * 14)
    f = rules.apply(tl(s))
    assert [x for x, _ in types(f)] == ["T1"], types(f)
    assert abs(f[0].t - 6.0) < 0.6, f[0].t


def test_normal_leave_closed_safe():
    n = 30
    s = make(n, cashier_in_room=[1.0] * 15 + [0.0] * 15, cashier_at_workplace=[1.0] * 12 + [0.0] * 18)
    assert rules.apply(tl(s)) == []


def test_phone():
    n = 20
    s = make(n, cashier_hands=["empty"] * 8 + ["phone"] * 6 + ["empty"] * 6)
    assert [x for x, _ in types(rules.apply(tl(s)))] == ["T4"]


def test_no_id_before_cash():
    n = 40
    s = make(n, visitors=[0.0] * 4 + [1.0] * 30 + [0.0] * 6)
    ev = [{"t": 5.0, "event": "cash_given_to_cashier", "actor": "client"}]
    assert "T9" in [x for x, _ in types(rules.apply(tl(s, ev)))]


def test_id_then_cash_is_normal():
    n = 40
    s = make(n, visitors=[0.0] * 4 + [1.0] * 30 + [0.0] * 6)
    ev = [{"t": 3.0, "event": "id_card_given_to_cashier", "actor": "client"},
          {"t": 6.0, "event": "cash_given_to_cashier", "actor": "client"}]
    assert rules.apply(tl(s, ev)) == []


def test_entry_with_cash_on_desk():
    n = 30
    s = make(n, visitors=[0.0] * 10 + [1.0] * 20, cash_on_desk=[1.0] * 30)
    assert "T2" in [x for x, _ in types(rules.apply(tl(s)))]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
