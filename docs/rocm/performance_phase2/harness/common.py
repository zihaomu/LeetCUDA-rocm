import collections
import math
import statistics
import torch


def warmup(ours, baseline, milliseconds=200):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(8):
            ours()
            baseline()
    with torch.cuda.stream(stream):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        while True:
            for _ in range(16):
                graph.replay()
            end.record()
            end.synchronize()
            if start.elapsed_time(end) >= milliseconds:
                break


def gates(rows, key):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    unstable = []
    for case, values in groups.items():
        for field in ('ours_ms', 'baseline_ms'):
            v = [r[field] for r in values]
            spread = (max(v) - min(v)) / statistics.median(v)
            if spread > .05:
                unstable.append(dict(case=case, side=field, spread=spread))
    gain = math.exp(statistics.mean(math.log(r['baseline_over_ours']) for r in rows))
    worst = max(r['ours_ms'] / r['baseline_ms'] for r in rows)
    return dict(geomean_speedup=gain, worst_slowdown=worst, unstable=unstable,
                passed=gain >= 1.05 and worst <= 1.1 and not unstable,
                stability_passed=not unstable, regression_passed=worst <= 1.1)
