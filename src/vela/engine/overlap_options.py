"""Architecture overlap knobs layered on top of the fast backend.

READ_PORTS=k models k independent arena read port groups (full crossbar
assumption, steady-state round-robin assignment in emission order). Width and
DMA knobs live in base_engine.py behind environment flags. All knobs default
off; the unmodified path reproduces the v14/v15 baseline bit-exactly.
"""
import os


def install(module):
    k = int(os.environ.get("READ_PORTS", "1"))
    if k <= 1:
        return
    Base = module.Scheduler

    class PortScheduler(Base):
        def __init__(self, cfg, banks):
            super().__init__(cfg, banks)
            self.bank_resources = dict(self.bank_resources)
            for i in range(1, k):
                name = "arena_read_p%d" % i
                self.bank_resources[name] = tuple("%s:%d" % (name, j) for j in range(banks))
            self._rp = 0

        def add(self, cat, duration, deps=(), resource=None, **kw):
            if resource == "arena_read":
                resource = "arena_read_p%d" % self._rp
                self._rp = (self._rp + 1) % k
            return super().add(cat, duration, deps, resource, **kw)

    module.Scheduler = PortScheduler
