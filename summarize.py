import json, glob, csv, math, shutil
from pathlib import Path
from typing import List

ROOT = Path("results")
OUT_CSV = ROOT / "summary.csv"
PREFERRED_KEYS = ["acc", "acc_norm", "exact_match", "macro_f1"]

def pick_metric(metrics: dict):
    # Try preferred keys first
    for k in PREFERRED_KEYS:
        v = metrics.get(k, None)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return v, k
    # Fallback: first numeric value
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return v, k
    return None, None

rows = []

# Grab any results_*.json files anywhere under results/
for jf in glob.glob(str(ROOT / "**" / "results_*.json"), recursive=True):
    jf_path = Path(jf)
    model_name = jf_path.parent.name  # e.g., meta-llama_Llama-3.1-8B-Instruct or phase subdir parent
    # If you have phase subfolders (e.g., results/<model>/phase1/...),
    # put the model as the parent of the phase:
    if model_name.startswith("phase") and jf_path.parent.parent.name:
        model_name = jf_path.parent.parent.name

    with open(jf_path, "r") as f:
        data = json.load(f)

    # Some versions save results under "results"
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], dict):
        task_map = data["results"]
    else:
        task_map = data  # assume old/simple format

    if not isinstance(task_map, dict):
        # Skip unexpected formats
        continue

    for task, metrics in task_map.items():
        if not isinstance(metrics, dict):
            # Skip non-dict entries (e.g., "date": "...")
            continue
        val, key_used = pick_metric(metrics)
        if val is None:
            continue
        rows.append({
            "model": model_name,
            "task": task,
            "metric": key_used,
            "score": val,
            "file": jf_path.as_posix(),
        })

# Sort for nice viewing: group by task first, then by model
# Custom task order: aime24 first, ifeval second, others alphabetically
TASK_ORDER = ["aime24", "ifeval"]
def task_sort_key(task):
    if task in TASK_ORDER:
        return (TASK_ORDER.index(task), task)
    return (len(TASK_ORDER), task)

rows.sort(key=lambda r: (task_sort_key(r["task"]), r["model"]))

# Write CSV
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "task", "metric", "score", "file"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"\n✅ Summary written to: {OUT_CSV}")

# Also print a Markdown table
def _clean_metric_name(metric: str) -> str:
    """Strip off the ',none' suffix from metric names for cleaner display."""
    if "," in metric:
        return metric.split(",")[0]
    return metric


BENCHMARK_DESCRIPTIONS = {
    "aime24": "Math problem solving",
    "ifeval": "Instruction following",
}


def _get_benchmark_description(task: str) -> str:
    """Get a human-readable description of what a benchmark tests."""
    return BENCHMARK_DESCRIPTIONS.get(task, "General evaluation")


def _format_score(s):
    return f"{s:.3f}" if isinstance(s, float) else str(s)


def _score_interpretation(score, metric):
    """Return a human-readable interpretation of the score."""
    if not isinstance(score, (int, float)):
        return ""
    
    # For exact_match and accuracy-like metrics (0-1 scale)
    if metric in ["exact_match", "acc", "acc_norm", "macro_f1"]:
        pct = score * 100
        return f"{pct:.1f}% correct"
    
    # For other metrics, just show percentage
    if "acc" in metric.lower() or "match" in metric.lower():
        pct = score * 100
        return f"{pct:.1f}% success"
    
    # Default: just return the raw value
    return f"{score:.3f}"


def print_pretty_table(rows: List[dict]):
    """Print a nice table to the terminal.

    Preference order:
    1. rich (best)
    2. tabulate
    3. simple manual table fallback
    """
    # Try rich first
    try:
        from rich.table import Table
        from rich.console import Console

        console = Console()
        table = Table(show_header=True, header_style="bold", row_styles=[""], show_lines=True)
        table.add_column("Model", overflow="fold")
        table.add_column("Task", overflow="fold")
        table.add_column("Description", overflow="fold")
        table.add_column("Metric", no_wrap=True)  # Don't wrap or truncate metric names
        table.add_column("Score", justify="right")
        table.add_column("Result", overflow="fold")

        for r in rows:
            interp = _score_interpretation(r["score"], r["metric"])
            clean_metric = _clean_metric_name(r["metric"])
            description = _get_benchmark_description(r["task"])
            table.add_row(r["model"], r["task"], description, clean_metric, _format_score(r["score"]), interp)

        console.print(table)
        return
    except Exception:
        pass

    # Try tabulate next
    try:
        from tabulate import tabulate

        table = [
            [r["model"], r["task"], _get_benchmark_description(r["task"]), _clean_metric_name(r["metric"]), _format_score(r["score"]), _score_interpretation(r["score"], r["metric"])]
            for r in rows
        ]
        print(tabulate(table, headers=["Model", "Task", "Description", "Metric", "Score", "Result"], tablefmt="github"))
        return
    except Exception:
        pass

    # Manual fallback: compute column widths, cap them so lines don't explode
    if not rows:
        print("No rows to display")
        return

    # Terminal width helps pick sensible truncation
    term_width = shutil.get_terminal_size((200, 20)).columns

    max_model = min(max((len(r["model"]) for r in rows), default=5), 30)
    max_task = min(max((len(r["task"]) for r in rows), default=4), 15)
    max_desc = min(max((len(_get_benchmark_description(r["task"])) for r in rows), default=10), 25)
    max_metric = max((len(_clean_metric_name(r["metric"])) for r in rows), default=10)  # Don't truncate metric names
    max_result = min(max((len(_score_interpretation(r["score"], r["metric"])) for r in rows), default=10), 20)

    hdr_model = "Model"
    hdr_task = "Task"
    hdr_desc = "Description"
    hdr_metric = "Metric"
    hdr_score = "Score"
    hdr_result = "Result"

    fmt = f"{{:<{max_model}}}  {{:<{max_task}}}  {{:<{max_desc}}}  {{:<{max_metric}}}  {{:>7}}  {{:<{max_result}}}"
    print(fmt.format(hdr_model, hdr_task, hdr_desc, hdr_metric, hdr_score, hdr_result))
    print("-" * min(term_width, max_model + max_task + max_desc + max_metric + max_result + 40))

    for r in rows:
        model = r["model"]
        task = r["task"]
        desc = _get_benchmark_description(task)
        metric = _clean_metric_name(r["metric"])
        if len(model) > max_model:
            model = model[: max_model - 1] + "…"
        if len(task) > max_task:
            task = task[: max_task - 1] + "…"
        if len(desc) > max_desc:
            desc = desc[: max_desc - 1] + "…"
        interp = _score_interpretation(r["score"], r["metric"])
        if len(interp) > max_result:
            interp = interp[: max_result - 1] + "…"
        print(fmt.format(model, task, desc, metric, _format_score(r["score"]), interp))
        print("-" * min(term_width, max_model + max_task + max_desc + max_metric + max_result + 40))


print("\nEvaluation Results:\n")
print_pretty_table(rows)