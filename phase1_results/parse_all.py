"""Parse every *.log in /workspace/phase1_results into a matching *_results.json."""
import re, json, pathlib

results_dir = pathlib.Path("/workspace/phase1_results")
for log_path in sorted(results_dir.glob("*.log")):
    if log_path.stat().st_size == 0:
        continue

    log = log_path.read_text()
    lines = [l for l in log.splitlines() if l.startswith("|") and "---" not in l]
    results = {}
    current_task = None
    for line in lines:
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 7:
            continue
        task, _, _, _, metric, _, value = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]
        if task and task != "Tasks":
            current_task = task
        if current_task and metric and value:
            try:
                results.setdefault(current_task, {})[metric] = float(value)
            except ValueError:
                pass

    ppl_match = re.search(r"\[INFO\] Perplexity:\s+([\d.]+)", log)
    if ppl_match:
        results.setdefault("_wikitext", {})["perplexity"] = float(ppl_match.group(1))

    if results:
        out_path = log_path.with_name(log_path.stem + "_results.json")
        out_path.write_text(json.dumps(results, indent=2))
        print(f"{log_path.name:42s} -> {out_path.name} ({sum(len(v) for v in results.values())} values)")
    else:
        print(f"{log_path.name:42s} -> no parseable results yet")
