import numpy as np
from config import TX_COST, TSL_THRESHOLD, TSL_WINDOW

class Portfolio:
    def __init__(self):
        self.current = None
        self.returns = []
        self.in_cash = False

    def update(self, r):
        self.returns.append(r)

    def tsl_trigger(self):
        if len(self.returns) < TSL_WINDOW:
            return False
        return sum(self.returns[-TSL_WINDOW:]) < TSL_THRESHOLD

    def decide(self, scores):
        best = max(scores, key=scores.get)

        if self.tsl_trigger():
            self.in_cash = True
            return "CASH", 0

        if self.current and self.current != best:
            scores[best] -= TX_COST

        self.current = best
        self.in_cash = False

        return best, scores[best]
