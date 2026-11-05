"""Paired fixed-size HIP graphs with alternating event measurements."""

import math
import statistics

import torch


def graph_pair(ours, baseline, calls=32, sample_ms=40, trials=5):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs, repetitions = {}, {}
    with torch.cuda.stream(stream):
        for name, fn in (("ours", ours), ("baseline", baseline)):
            for _ in range(5):
                fn()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(calls):
                    fn()
            for _ in range(8):
                graph.replay()
            start, stop = (
                torch.cuda.Event(enable_timing=True) for _ in range(2)
            )
            start.record()
            for _ in range(8):
                graph.replay()
            stop.record()
            stop.synchronize()
            estimate = start.elapsed_time(stop) / 8
            repetitions[name] = min(
                4096, max(1, math.ceil(sample_ms / max(estimate, 1e-6)))
            )
            graphs[name] = graph
        samples = {"ours": [], "baseline": []}
        for trial in range(trials):
            order = (
                ("ours", "baseline") if trial % 2 == 0 else ("baseline", "ours")
            )
            for name in order:
                graph = graphs[name]
                start, stop = (
                    torch.cuda.Event(enable_timing=True) for _ in range(2)
                )
                graph.replay()
                start.record()
                for _ in range(repetitions[name]):
                    graph.replay()
                stop.record()
                stop.synchronize()
                samples[name].append(
                    start.elapsed_time(stop) / (repetitions[name] * calls)
                )
    a, b = (statistics.median(samples[name]) for name in ("ours", "baseline"))
    return {
        "ours_ms": a,
        "baseline_ms": b,
        "baseline_over_ours": b / a,
        "samples_ms": samples,
        "calls_per_graph": calls,
        "replays_per_sample": repetitions,
    }
