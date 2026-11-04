"""A first, deliberately simple RL trigger for DWAFP.

The policy decides only WHEN to request impact analysis (WAIT or ANALYZE) for an executing task.
It never changes assignments or queues: it reads frozen snapshots and asks the analysis/repair
layer through an interface (see interface.py).
"""
