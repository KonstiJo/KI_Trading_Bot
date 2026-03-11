"""Demonstration des Gewinnfaktors und seines mathematischen Maximums.

Der Gewinnfaktor (profit factor) ist definiert als Summe der
Gewinne geteilt durch Summe der Verluste. Rein mathematisch kann dieser
Wert beliebig groß werden, wenn die Verlustseite gegen Null geht.

Dieses kleine Modul enthält einen Helfer, um den Profitfactor für eine
gegebene Liste von Trade-Ergebnissen zu berechnen, und zeigt anhand
von Beispielen, warum es kein finite Maximum gibt.
"""

from typing import Iterable


def profit_factor(gains: Iterable[float], losses: Iterable[float]) -> float:
    """Berechnet den Profitfactor: Summe(gains)/Summe(abs(losses)).
    Wenn es keine Verluste gibt, wird `float('inf')` zurückgegeben.
    """
    total_gain = sum(gains)
    total_loss = sum(abs(l) for l in losses)
    if total_loss == 0:
        return float("inf")
    return total_gain / total_loss


if __name__ == "__main__":
    print("== running highest_profit_factor.py ==")
    # ein paar Beispiele
    print("Beispiel 1")
    print("Gewinne: [100, 50], Verluste: [-25, -25]")
    print("Profitfaktor:", profit_factor([100, 50], [-25, -25]))

    print("\nBeispiel 2 – kein Verlust")
    print("Gewinne: [100, 50], Verluste: []")
    print("Profitfaktor:", profit_factor([100, 50], []))

    print("\nMathematisch gesehen lässt sich der Profitfaktor beliebig"\
          " steigern, indem man die Summe der Verluste gegen 0 schrumpfen"\
          " lässt.")
