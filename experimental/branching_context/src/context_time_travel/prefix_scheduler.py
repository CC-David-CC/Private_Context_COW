"""Bounded-wait, backlog-weighted admission. Physical placement stays in COW.

This is an admission heuristic, not an optimal-throughput claim. Already running
native batches are never migrated or interrupted by queue reprioritization.
"""
from collections import Counter

MAX_WEIGHTED_WAIT_SECONDS = 2.0


def order_jobs(jobs, active, *, now):
    jobs = list(jobs)
    counts = Counter(job.prefill_key for job in jobs)
    def rank(pair):
        index, job = pair
        age = max(0., now - job.request.created)
        if age >= MAX_WEIGHTED_WAIT_SECONDS:
            return (0, job.request.created, index)
        weight = counts[job.prefill_key] / (1 + active.get(job.prefill_key, 0))
        return (1, -weight, job.request.created, index)
    return [job for _, job in sorted(enumerate(jobs), key=rank)]


def select_worker(eligible, warm, seconds_per_token):
    # Keep a valid resident image when possible. With equivalent residency,
    # measured worker service rate breaks ties; unmeasured workers retain the
    # configured stable order. No native lease is touched while routing.
    candidates = [gpu for gpu in eligible if gpu in warm] or list(eligible)
    if not candidates:
        raise ValueError('No eligible execution slot')
    return min(candidates, key=lambda gpu: seconds_per_token.get(gpu, 0.))
