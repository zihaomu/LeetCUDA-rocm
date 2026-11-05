"""Append a timestamped migration result to both requested Markdown locations."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--message", required=True)
    parser.add_argument("--current")
    args = parser.parse_args()
    source = ROOT / "docs/rocm/leetcuda_rocm10_live_progress.md"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = source.read_text().splitlines()
    inventory = json.loads(
        (source.parent / "leetcuda_rocm10_operator_inventory.json").read_text()
    )
    summary = inventory["summary"]
    for i, line in enumerate(lines):
        if line.startswith("最近更新（UTC）："):
            lines[i] = f"最近更新（UTC）：{stamp}。任务状态：**执行中**。"
        if args.current and line.startswith("- 当前执行："):
            lines[i] = "- 当前执行：" + args.current
        if line.startswith("- 原接口："):
            lines[i] = (
                f"- 原接口：**{summary['qualified']}/229（{summary['qualified_percent']:.2f}%）**；"
                f"待迁移 **{summary['pending']}**，精度未验收 **{summary['numerical_rejected']}**。"
            )
    start = lines.index("## 尚未验收的逐接口清单")
    end = lines.index("## 实时实验记录")
    pending = [
        "## 尚未验收的逐接口清单",
        "",
        "| 模块 | 接口 | 正式状态 |",
        "| --- | --- | --- |",
    ]
    pending += [
        f"| {e['module']} | `{e['function']}` | {e['status']} |"
        for e in inventory["entries"]
        if e["status"] != "qualified"
    ]
    lines[start:end] = pending + [""]
    lines += ["", f"- **{stamp}**：{args.message}"]
    content = "\n".join(lines) + "\n"
    source.write_text(content)
    destination = ROOT.parent / "doc" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
    print(destination)


if __name__ == "__main__":
    main()
