# src/budget.py
from __future__ import annotations

class BudgetScheduler:
    """Converts a trajectory-level (soft) budget C over T steps into a per-step threshold b_t
    based on remaining budget.

    Typical usage:
        sched = BudgetScheduler(C=2.0, T=60)
        bt = sched.bt()
        ...
        sched.step(cost_t)   # cost_t in {0,1}
    """

    def __init__(self, C: float, T: int, min_bt: float = 0.0):
        self.C = float(C)
        self.T = int(T)
        self.min_bt = float(min_bt)
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.B_rem = float(self.C)

    def step(self, cost_t: float) -> None:
        self.B_rem = max(0.0, self.B_rem - float(cost_t))
        self.t += 1

    def bt(self) -> float:
        T_rem = max(1, self.T - self.t)
        return max(self.min_bt, self.B_rem / T_rem)
