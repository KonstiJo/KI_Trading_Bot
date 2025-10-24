import pytest
from bot import check_for_signal

def test_buy_signal():
    """Testet, ob ein Kaufsignal korrekt erkannt wird."""
    # Szenario: Fast EMA steigt von unter Slow EMA auf über Slow EMA
    fast_prev = 100; fast_cur = 105
    slow_prev = 102; slow_cur = 103
    in_position = False  # bisher keine Position
    signal = check_for_signal(fast_prev, fast_cur, slow_prev, slow_cur, in_position)
    assert signal == "BUY", "Es sollte ein BUY-Signal erzeugt werden, wenn der schnelle EMA den langsamen von unten kreuzt."

def test_sell_signal():
    """Testet, ob ein Verkaufssignal korrekt erkannt wird."""
    # Szenario: Fast EMA fällt von über Slow EMA auf unter Slow EMA
    fast_prev = 110; fast_cur = 105
    slow_prev = 108; slow_cur = 107
    in_position = True  # wir halten eine Position
    signal = check_for_signal(fast_prev, fast_cur, slow_prev, slow_cur, in_position)
    assert signal == "SELL", "Es sollte ein SELL-Signal erzeugt werden, wenn der schnelle EMA den langsamen von oben kreuzt."

def test_no_signal_when_no_cross():
    """Kein Signal, wenn keine Überkreuzung stattfindet."""
    # Szenario: EMAs bleiben in gleicher Reihenfolge (fast EMA schon über slow EMA und bleibt drüber)
    fast_prev = 200; fast_cur = 210
    slow_prev = 150; slow_cur = 160
    in_position = False
    signal = check_for_signal(fast_prev, fast_cur, slow_prev, slow_cur, in_position)
    assert signal is None, "Es sollte kein Signal geben, wenn kein Crossover passiert."
