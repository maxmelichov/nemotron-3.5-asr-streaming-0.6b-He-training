#!/usr/bin/env python3
"""Step 3 — Evaluate WER in cache-aware streaming mode (blog recipe).

Runs the ivrit.ai-style Hebrew benchmark suite by default:
  ivrit-ai/eval-d1, eval-whatsapp, SASpeech, FLEURS, Common Voice, hebrew_speech_kan

Usage:
  python scripts/build_eval_benchmarks.py          # once: cache eval manifests + wav
  python scripts/eval_streaming.py --compare-base
  python scripts/eval_streaming.py --benchmark fleurs saspeech
  python scripts/eval_streaming.py --test-manifest data/manifests/test.json  # legacy single manifest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import text_norm
from common import ensure_cuda_home, load_config, nemo_root, repo_root, run


def parse_wer(output: str) -> float:
    matches = re.findall(r"WER% of streaming mode:\s*([0-9.]+)", output)
    if not matches:
        raise RuntimeError("Could not parse WER from NeMo streaming infer output")
    return float(matches[-1])


def streaming_eval(
    infer_script: Path,
    model_path: Path,
    manifest: Path,
    target_lang: str,
    att: list[int],
    output_dir: Path,
) -> float:
    att_str = f"[{att[0]},{att[1]}]"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"python {infer_script} "
        f"model_path={model_path.resolve()} "
        f"dataset_manifest={manifest.resolve()} "
        f"output_path={output_dir.resolve()} "
        f"target_lang={target_lang} "
        f'att_context_size="{att_str}" '
        f"decoder_type=rnnt "
        f"pad_and_drop_preencoded=true "
        f"batch_size=16 "
        f"strip_lang_tags=true "
        f"cuda=0"
    )
    print("+", cmd, flush=True)
    # The infer script ships with NEMO_ROOT but `import nemo` resolves to whatever is
    # installed in the venv (an older /opt/NeMo on the training box). finetune.py already
    # pins PYTHONPATH to NEMO_ROOT for exactly this reason -- eval must match, or the
    # model evaluates under a different NeMo than it trained with (or crashes outright
    # on missing symbols, as it did with get_inference_device).
    env = os.environ.copy()
    # .../NeMo-main/examples/asr/asr_cache_aware_streaming/<script> -> NeMo-main
    env["PYTHONPATH"] = str(infer_script.parents[3]) + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    print(res.stdout[-3000:], flush=True)
    if res.returncode != 0:
        raise RuntimeError(f"Streaming infer failed (rc={res.returncode})")
    raw = parse_wer(res.stdout)
    norm = normalized_wer_from_output(output_dir)
    if norm is not None:
        print(f"  raw WER {raw:.2f}%  |  punctuation-insensitive WER {norm:.2f}%", flush=True)
    return raw, norm


def normalized_wer_from_output(output_dir: Path) -> float | None:
    """Re-score the infer script's own hypotheses ignoring punctuation and niqqud.

    NeMo reports WER against the raw reference, where an attached comma turns a
    correctly-heard word into an error. The script writes every (text, pred_text) pair
    to output_path, so we can rescore from that file rather than re-running inference.
    """
    hyp_files = sorted(output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not hyp_files:
        return None
    refs: list[str] = []
    hyps: list[str] = []
    with hyp_files[-1].open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "text" in rec and "pred_text" in rec:
                refs.append(rec["text"])
                hyps.append(rec["pred_text"])
    if not refs:
        return None
    rate, _, _ = text_norm.wer(refs, hyps, normalized=True)
    return rate * 100.0


def benchmark_manifests(cfg: dict, names: list[str] | None) -> list[tuple[str, Path, str]]:
    eval_cfg = cfg["evaluation"]
    manifest_dir = Path(eval_cfg["manifest_dir"])
    specs = eval_cfg["benchmarks"]
    chosen = names or [k for k, v in specs.items() if v.get("enabled", True)]
    out: list[tuple[str, Path, str]] = []
    for name in chosen:
        spec = specs.get(name)
        if spec is None:
            raise KeyError(f"Unknown benchmark: {name}")
        manifest = manifest_dir / f"{name}.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Eval manifest missing: {manifest}\nRun: python scripts/build_eval_benchmarks.py"
            )
        label = spec.get("description") or name
        out.append((name, manifest, label))
    return out


def ensure_benchmark_manifests(cfg: dict, names: list[str] | None) -> None:
    eval_cfg = cfg["evaluation"]
    manifest_dir = Path(eval_cfg["manifest_dir"])
    specs = eval_cfg["benchmarks"]
    chosen = names or [k for k, v in specs.items() if v.get("enabled", True)]
    missing = [n for n in chosen if not (manifest_dir / f"{n}.json").exists()]
    if missing:
        run(f"uv run {repo_root() / 'scripts/build_eval_benchmarks.py'} --benchmark {' '.join(missing)}")


def print_results_table(
    results: dict[str, dict[str, float]],
    headline_att: str,
    compare_base: bool,
) -> None:
    print("\n" + "=" * 88)
    print(f"RAW WER (%) — cache-aware streaming, att_context_size={headline_att}")
    print("=" * 88)
    if compare_base and "base" in results and "finetuned" in results:
        print(f"| {'Benchmark':<22} | {'Base WER':>10} | {'Fine-tuned WER':>14} | {'Rel. improvement':>16} |")
        print(f"|{'-'*24}|{'-'*12}|{'-'*16}|{'-'*18}|")
        for bench in results["finetuned"]:
            b = results["base"].get(bench)
            f = results["finetuned"][bench]
            if b is None:
                continue
            rel = (b - f) / b * 100 if b else 0.0
            print(f"| {bench:<22} | {b:>9.1f}% | {f:>13.1f}% | {rel:>15.1f}% |")
    else:
        tag = "finetuned" if "finetuned" in results else next(iter(results))
        print(f"| {'Benchmark':<22} | {'WER':>10} |")
        print(f"|{'-'*24}|{'-'*12}|")
        for bench, wer in results[tag].items():
            print(f"| {bench:<22} | {wer:>9.1f}% |")
    print("=" * 88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--model", type=Path, default=Path("checkpoints/hebrew-finetuned.nemo"))
    ap.add_argument("--base-model", type=Path, default=Path("checkpoints/nemotron-3.5-asr-base.nemo"))
    ap.add_argument("--test-manifest", type=Path, help="Single manifest (legacy mode; skips benchmark suite)")
    ap.add_argument(
        "--benchmark",
        nargs="*",
        help="Eval benchmarks to run (default: all enabled in config.yaml)",
    )
    ap.add_argument("--compare-base", action="store_true", help="Also eval base model")
    ap.add_argument("--ladder", action="store_true", help="Sweep all latency settings")
    ap.add_argument("--att-context", type=str, help='e.g. "[56,0]"')
    ap.add_argument("--build-benchmarks", action="store_true", help="Build missing eval manifests before eval")
    args = ap.parse_args()

    ensure_cuda_home()
    cfg = load_config(args.config)
    target_lang = cfg["project"]["target_lang"]
    eval_cfg = cfg["evaluation"]

    if args.build_benchmarks or not args.test_manifest:
        ensure_benchmark_manifests(cfg, args.benchmark)

    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}\nRun: python scripts/finetune.py")

    nemo = nemo_root()
    infer_script = nemo / "examples" / "asr" / "asr_cache_aware_streaming" / "speech_to_text_cache_aware_streaming_infer.py"

    if args.ladder:
        att_list = [item[0] for item in eval_cfg["latency_ladder"]]
    elif args.att_context:
        att_list = [json.loads(args.att_context.replace(" ", ""))]
    else:
        att_list = [eval_cfg["att_context_size"]]

    models = [("finetuned", args.model)]
    if args.compare_base:
        if not args.base_model.exists():
            run(f"uv run {repo_root() / 'scripts/download_model.py'} --output {args.base_model}")
        models.insert(0, ("base", args.base_model))

    if args.test_manifest:
        eval_items = [("custom", args.test_manifest, "custom test manifest")]
    else:
        eval_items = benchmark_manifests(cfg, args.benchmark)

    results: dict[str, dict[str, float]] = {tag: {} for tag, _ in models}
    # Punctuation/niqqud-insensitive scores, reported alongside raw. 20.9% of dev
    # reference words carry punctuation, and WER charges a full word error for a missing
    # comma -- so raw WER measures punctuation as much as recognition.
    norm_results: dict[str, dict[str, float]] = {}

    for bench_name, manifest, _label in eval_items:
        for tag, model_path in models:
            for att in att_list:
                att_key = f"[{att[0]}, {att[1]}]"
                out_dir = Path("exp") / "eval" / tag / bench_name / f"att_{att[0]}_{att[1]}"
                try:
                    wer, norm_wer = streaming_eval(
                        infer_script, model_path, manifest, target_lang, att, out_dir
                    )
                except Exception as exc:
                    print(f"FAIL [{tag}] {bench_name} att={att_key}: {exc}", flush=True)
                    continue
                key = bench_name if len(att_list) == 1 else f"{bench_name} {att_key}"
                results[tag][key] = wer
                if norm_wer is not None:
                    norm_results.setdefault(tag, {})[key] = norm_wer
                print(
                    f"[{tag}] {bench_name} att={att_key} RAW WER = {wer:.2f}%"
                    + (f", NORMALIZED = {norm_wer:.2f}%" if norm_wer is not None else ""),
                    flush=True,
                )

    headline_att = f"[{eval_cfg['att_context_size'][0]}, {eval_cfg['att_context_size'][1]}]"
    if len(att_list) == 1:
        print_results_table(results, headline_att, args.compare_base)
        if norm_results:
            print("\nPunctuation/niqqud-insensitive (words only):")
            print_results_table(norm_results, headline_att, args.compare_base)

    out_json = Path("exp/eval/streaming_result.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"raw": results, "normalized": norm_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults -> {out_json}")


if __name__ == "__main__":
    main()
