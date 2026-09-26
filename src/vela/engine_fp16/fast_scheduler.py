"""Pure Python acceleration; preserves event dictionaries and scheduling rules.

No compiled extension or additional dependency. See tests/test_scheduler.py before
porting to a model with different event semantics.
"""
from array import array
from functools import lru_cache
import sys


def make_scheduler(reference):
    class FastScheduler(reference):
        def __init__(self, cfg, banks):
            super().__init__(cfg, banks)
            self.end_times = array("q")
            self.expansion_cache = {}

        def end(self, index):
            return self.end_times[index] if index >= 0 else 0

        def add(
            self,
            cat,
            duration,
            deps=(),
            resource=None,
            reason="",
            read=0,
            write=0,
            onchip=0,
            traffic="",
        ):
            if duration < 0:
                raise ValueError("negative duration")
            signature = (
                (cat,)
                if resource is None
                else ((resource,) if isinstance(resource, str) else tuple(resource))
            )
            resources = self.expansion_cache.get(signature)
            if resources is None:
                expanded = []
                for res in signature:
                    expanded.extend(self.bank_resources.get(res, (res,)))
                expanded = tuple(expanded)
                resources = self.resource_sets.setdefault(expanded, expanded)
                self.expansion_cache[signature] = resources
            ends = self.end_times
            dep = set(i for i in deps if i is not None and i >= 0)
            if not self.cfg.overlap and self.serial >= 0:
                dep.add(self.serial)
            ready = 0
            for i in dep:
                if ends[i] > ready:
                    ready = ends[i]
            last = self.last
            dep.update(last[r] for r in resources if r in last)
            start = ready
            for i in dep:
                if ends[i] > start:
                    start = ends[i]
            blockers = [i for i in dep if ends[i] == start] if start else []
            ident = len(self.events)
            end = start + int(duration)
            event = dict(
                id=ident,
                op=self.now_op,
                stage=self.stage,
                module=self.module,
                category=cat,
                start=start,
                end=end,
                ready=ready,
                resource_wait=start - ready,
                predecessors=sorted(dep),
                critical_predecessors=blockers,
                resources=resources,
                reason=sys.intern(reason),
                read_bytes=int(read),
                write_bytes=int(write),
                onchip_bytes=int(onchip),
                traffic=traffic,
            )
            self.events.append(event)
            ends.append(end)
            for res in resources:
                last[res] = ident
            self.serial = ident
            return ident

    return FastScheduler


@lru_cache(maxsize=8192)
def gather_pages(ranges):
    # Preserve first occurrence order, including ties in dependency selection.
    pages = dict.fromkeys(
        p for offset, size in ranges for p in range(offset // 1024, (offset + size + 1023) // 1024)
    )
    return tuple(pages), sum(size for offset, size in ranges)


def gather_ranges(self, key, ranges, base):
    ranges = tuple(ranges)
    duration, resources, words = self.gather_pattern(ranges, self.cache[key]["offset"])
    pages, nbytes = gather_pages(ranges)
    records = self.hazards.setdefault(key, {})
    # Every range belongs to one read event; each page needs its prior writer
    # checked once. No new writer appears between these range checks.
    for page in pages:
        writer, _ = records.get(page, (-1, -1))
        base = self.end(base, writer)
    end = self.emit(
        "local_move",
        duration,
        base,
        resource=resources,
        reason="coalesced 64-bit word gather with bank conflicts",
        onchip=nbytes,
    )
    for page in pages:
        writer, reader = records.get(page, (-1, -1))
        records[page] = (writer, self.end(reader, end))
    self.last_access[key] = self.end(end, self.last_access.get(key, -1))
    self.stats["gather_words"] += words
    return end


def latest_event(self, *ids):
    """Same stable tie-breaking as max(ids, key=scheduler.end, default=-1)."""
    ends = self.s.end_times
    best, best_time = -1, -1
    for ident in ids:
        time = ends[ident] if ident >= 0 else 0
        if time > best_time:
            best, best_time = ident, time
    return best


def install_engine(module):
    """Explicit backend choice at entry; baseline remains available."""

    class FastEngine(module.Engine):
        gather_ranges = globals()["gather_ranges"]
        end = latest_event

    module.Scheduler = make_scheduler(module.Scheduler)
    module.Engine = FastEngine
