"""Measured restore-plus-work placement, inspired by Dynamo/KVCR routing.

Rates are estimates from this process, not a claim about future latency. Missing
measurements preserve stable warm affinity. Eligibility remains COW's decision.
"""
from dataclasses import dataclass
import math
import threading


@dataclass(frozen=True)
class Placement:
    worker: object
    reason: str
    estimated_seconds: float | None = None


class TransferCosts:
    def __init__(self):
        self._forward={}; self._restore={}; self._lock=threading.RLock()

    def _observe(self,rates,worker,units,seconds):
        if (type(units) is not int or units<=0 or type(seconds) not in (int,float)
                or not math.isfinite(seconds) or seconds<0):
            raise ValueError('Invalid placement measurement')
        sample=seconds/units
        with self._lock:rates[worker]=.75*rates.get(worker,sample)+.25*sample

    def observe_forward(self,worker,tokens,seconds):self._observe(self._forward,worker,tokens,seconds)
    def observe_restore(self,worker,nbytes,seconds):self._observe(self._restore,worker,nbytes,seconds)

    def choose(self,eligible,warm,nbytes,tokens):
        eligible=list(eligible); warm=set(warm)
        if (not eligible or type(nbytes) is not int or nbytes<0
                or type(tokens) is not int or tokens<0):
            raise ValueError('Invalid placement candidates or work extent')
        with self._lock:
            if any(w not in self._forward or (w not in warm and nbytes and w not in self._restore)
                   for w in eligible):
                return Placement(next((w for w in eligible if w in warm),eligible[0]),'unmeasured')
            estimates={w:tokens*self._forward[w]+(0 if w in warm or not nbytes
                       else nbytes*self._restore[w]) for w in eligible}
            worker=min(eligible,key=estimates.__getitem__)
            return Placement(worker,'measured',estimates[worker])

    def metrics(self):
        with self._lock:return dict(forward=dict(self._forward),restore=dict(self._restore))
