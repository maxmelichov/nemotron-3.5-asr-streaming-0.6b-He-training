#!/usr/bin/env python3
"""Step 2 — Fine-tune Nemotron 3.5 ASR on Hebrew data.

Follows the NVIDIA blog recipe:
  https://huggingface.co/blog/nvidia/fine-tuning-nemotron-35-asr
  speech_to_text_finetune.py + fastconformer_transducer_bpe_streaming_prompt.yaml
  init_from_nemo_model, step budget, bf16, target_lang=he-IL prompt conditioning

Telephony (phone-call) training uses NeMo online augmentations by default:
  G.711 / AMR codec, bandpass, gain, noise, slight speed/shift perturbations.

Usage:
  python scripts/build_noise_manifest.py --noise-dir /path/to/noise   # optional
  python scripts/finetune.py
  python scripts/finetune.py --no-augment                             # disable augmentation
  python scripts/finetune.py --max-steps 2000 --limit-train-batches 500
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    augmentor_hydra_overrides,
    compute_train_steps_per_epoch,
    ensure_cuda_home,
    consolidate_exp_checkpoints,
    filter_manifest_by_duration,
    find_best_nemo,
    find_latest_nemo,
    read_checkpoint_resume_info,
    hebrew_only_enabled,
    load_config,
    manifest_hours,
    nemo_root,
    read_nemo_manifest,
    register_custom_perturbations,
    repo_root,
    resolve_augmentor_cfg,
    run,
    run_training,
    setup_training_env,
    validate_hebrew_manifest,
    write_nemo_manifest,
    _hydra_value,
)


def ensure_train_deps() -> None:
    """NeMo + lightning are installed via uv pip (not uv sync) — verify before training."""
    missing: list[str] = []
    try:
        import lightning.pytorch  # noqa: F401
    except ImportError:
        missing.append("lightning")
    try:
        import nemo  # noqa: F401
    except ImportError:
        missing.append("nemo_toolkit[asr]")
    if not missing:
        return
    sys.exit(
        "Missing training packages: " + ", ".join(missing) + "\n"
        "Run: ./setup.sh\n"
        "Or: uv pip install lightning "
        '"nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(repo_root() / "config.yaml"))
    ap.add_argument("--base-model", type=Path, help=".nemo checkpoint (default: Hebrew ckpt if present, else HF base)")
    ap.add_argument("--from-scratch", action="store_true", help="Init from multilingual HF base, ignore saved Hebrew .nemo")
    ap.add_argument("--latest-checkpoint", action="store_true", help="Init from most recent .nemo instead of best val_wer")
    ap.add_argument("--no-resume", action="store_true", help="Load best .nemo weights only (reset step/LR); default is full resume from best.ckpt")
    ap.add_argument("--resume-checkpoint", type=Path, help="Resume full training state from this .ckpt (optimizer, LR schedule, global_step)")
    ap.add_argument("--train-manifest", type=Path)
    ap.add_argument("--dev-manifest", type=Path)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--warmup", type=int,
                    help="NoamAnnealing warmup steps. Peak LR scales as lr*d_model^-0.5/sqrt(warmup), "
                         "so a longer warmup also lowers the peak")
    ap.add_argument("--batch-duration", type=int)
    ap.add_argument("--grad-accum", type=int, help="accumulate_grad_batches (effective batch = batch_duration x this x devices)")
    ap.add_argument("--devices", type=int)
    ap.add_argument("--val-interval", type=int, help="Validate every N optimizer steps (config default: 100)")
    ap.add_argument("--att-context", help='Encoder cache-aware context, e.g. "[56,13]"')
    ap.add_argument("--no-early-stopping", action="store_true", help="Disable the early-stopping callback")
    ap.add_argument("--limit-train-batches", type=int, help="Cap training epoch length (optimizer steps, not micro-batches)")
    ap.add_argument("--tarred", action="store_true", help="Use tarred train shards from config")
    ap.add_argument("--no-augment", action="store_true", help="Disable train-time audio augmentation")
    ap.add_argument("--exp-dir", type=Path, default=Path("exp"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_cuda_home()
    ensure_train_deps()
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    sample_rate = data_cfg["sample_rate"]
    target_lang = cfg["project"]["target_lang"]
    hebrew_only = hebrew_only_enabled(cfg)

    manifest_dir = Path(data_cfg["out_dir"]) / "manifests"
    train_manifest = args.train_manifest or manifest_dir / "train.json"
    dev_manifest = args.dev_manifest or manifest_dir / "dev.json"

    if not train_manifest.exists():
        sys.exit(f"Train manifest missing: {train_manifest}\nRun: python scripts/build_dataset.py")
    if not dev_manifest.exists():
        sys.exit(f"Dev manifest missing: {dev_manifest}\nRun: python scripts/build_dataset.py")

    train_rows = read_nemo_manifest(train_manifest)
    dev_rows = read_nemo_manifest(dev_manifest)
    if hebrew_only:
        validate_hebrew_manifest(train_rows, target_lang, train_manifest)
        validate_hebrew_manifest(dev_rows, target_lang, dev_manifest)
        print(f"Hebrew-only training: target_lang={target_lang}, prompt_mode=langID")

    max_clip_duration = train_cfg.get("max_clip_duration", 39.99)
    raw_train_count = len(train_rows)
    train_rows, skipped_long = filter_manifest_by_duration(train_rows, max_clip_duration)
    if skipped_long:
        filtered_manifest = manifest_dir / f"train.max{max_clip_duration:g}s.json"
        if not args.dry_run:
            write_nemo_manifest(train_rows, filtered_manifest, target_lang)
        train_manifest = filtered_manifest
        print(
            f"Excluded {skipped_long:,} train clips > {max_clip_duration}s "
            f"({raw_train_count:,} -> {len(train_rows):,}, {manifest_hours(train_rows):.1f} h)"
        )
    print(f"Train manifest: {len(train_rows):,} clips, {manifest_hours(train_rows):.1f} h")

    exp_name = train_cfg.get("exp_name", "hebrew_ft")
    save_top_k = train_cfg.get("save_top_k", 3)
    ckpt_sync_delay = train_cfg.get("checkpoint_sync_delay_sec", 180)
    init_from_best = train_cfg.get("init_from_best", True) and not args.latest_checkpoint

    base_nemo = args.base_model
    if base_nemo is None and not args.from_scratch:
        try:
            if init_from_best:
                base_nemo = find_best_nemo(args.exp_dir, exp_name, k=save_top_k, dry_run=args.dry_run)
            else:
                base_nemo = find_latest_nemo(args.exp_dir)
                print(f"Init from latest checkpoint: {base_nemo}")
        except FileNotFoundError:
            base_nemo = Path("checkpoints/nemotron-3.5-asr-base.nemo")
            print(f"No fine-tuned checkpoint found, using base: {base_nemo}")
    if base_nemo is None:
        base_nemo = Path("checkpoints/nemotron-3.5-asr-base.nemo")
    if not base_nemo.exists() and not args.dry_run:
        print(f"Base checkpoint not found at {base_nemo}, downloading ...")
        run(f"uv run {repo_root() / 'scripts/download_model.py'} --output {base_nemo}")
    elif args.dry_run and not base_nemo.exists():
        base_nemo = Path("checkpoints/nemotron-3.5-asr-base.nemo")

    link_dir = args.exp_dir / exp_name
    resume_ckpt: Path | None = None
    if args.resume_checkpoint:
        resume_ckpt = args.resume_checkpoint.resolve()
    elif (
        not args.from_scratch
        and not args.no_resume
        and train_cfg.get("resume_from_best", True)
        and init_from_best
    ):
        candidate = link_dir / "best.ckpt"
        if candidate.is_file():
            resume_ckpt = candidate.resolve()
        elif not args.dry_run:
            print("WARNING: resume_from_best enabled but best.ckpt missing — weights-only from .nemo")

    if resume_ckpt is not None:
        meta_path = link_dir / "best.json"
        wer_note = ""
        if meta_path.is_file():
            import json

            wer = float(json.loads(meta_path.read_text(encoding="utf-8"))["val_wer"])
            wer_note = f", val_wer={wer:.2%}"
        if not args.dry_run:
            info = read_checkpoint_resume_info(resume_ckpt)
            print(
                f"Full resume: {resume_ckpt}{wer_note}, "
                f"global_step={info['global_step']:,}, epoch={info['epoch']}"
            )
        else:
            print(f"Full resume: {resume_ckpt}{wer_note} (optimizer + LR schedule restored from ckpt)")

    register_custom_perturbations(cfg)

    nemo = nemo_root()
    finetune_script = nemo / "examples" / "asr" / "speech_to_text_finetune.py"
    finetune_entry = repo_root() / "scripts" / "nemo_finetune_entry.py"
    cfg_dir = nemo / "examples" / "asr" / "conf" / "fastconformer" / "cache_aware_streaming"
    cfg_name = "fastconformer_transducer_bpe_streaming_prompt"

    max_steps = args.max_steps if args.max_steps is not None else train_cfg.get("max_steps")
    lr = args.lr if args.lr is not None else train_cfg["lr"]
    warmup_steps = args.warmup if args.warmup is not None else train_cfg["warmup_steps"]
    batch_duration = args.batch_duration or train_cfg["batch_duration"]
    devices = args.devices or train_cfg["devices"]
    grad_accum = args.grad_accum or train_cfg.get("accumulate_grad_batches", 1)
    num_nodes = train_cfg.get("num_nodes", 1)
    num_workers = train_cfg.get("num_workers")
    limit_opt_steps = args.limit_train_batches
    if limit_opt_steps is None:
        limit_opt_steps = train_cfg.get("limit_train_batches")

    opt_steps_full = compute_train_steps_per_epoch(
        train_rows, batch_duration, grad_accum, devices, num_nodes
    )
    if limit_opt_steps is None:
        batch_note = (
            f"full manifest ({opt_steps_full:,} opt steps/epoch, "
            f"{opt_steps_full * grad_accum:,} micro-batches)"
        )
        micro_batches_per_epoch = opt_steps_full * grad_accum
    else:
        batch_note = f"limit_train_batches={limit_opt_steps:,} opt steps/epoch"
        micro_batches_per_epoch = limit_opt_steps * grad_accum
    opt_steps_per_epoch = limit_opt_steps if limit_opt_steps is not None else opt_steps_full
    step_note = f"max_steps={max_steps}" if max_steps is not None else "max_steps=NeMo default (500k)"
    val_interval_mode = train_cfg.get("val_interval_mode", "fraction")
    val_check_cfg = args.val_interval if args.val_interval is not None else train_cfg.get("val_check_interval", 0.5)
    if args.val_interval is not None:
        val_interval_mode = "optimizer_steps"
    val_every_epoch = train_cfg.get("val_every_epoch", False)
    val_every_epochs = None
    if val_interval_mode == "fraction" and not val_every_epoch:
        val_every = float(val_check_cfg)
        val_note = f"val every {val_every} of epoch (~{int(opt_steps_per_epoch * val_every):,} opt steps)"
    else:
        val_every_steps = int(val_check_cfg)
        val_every_micro = val_every_steps * grad_accum
        if val_every_epoch or val_every_micro > micro_batches_per_epoch:
            val_every = 1.0
            val_note = f"val every epoch (~{opt_steps_per_epoch:,} optimizer steps)"
        else:
            val_every = val_every_micro
            val_note = (
                f"val every {val_every_steps} optimizer steps "
                f"({val_every_micro} micro-batches)"
            )
    eff_per_gpu = batch_duration * grad_accum
    print(
        f"Training: {devices} GPU(s), micro-batch={batch_duration}s/GPU x grad_accum {grad_accum} "
        f"= {eff_per_gpu}s/GPU effective (global {eff_per_gpu * devices}s), "
        f"{batch_note}, {step_note}, {val_note}, lr={lr}"
    )
    # NoamAnnealing: lr_t = lr * d_model^-0.5 * min(t^-0.5, t*warmup^-1.5), so `lr` is a
    # scale factor, not the learning rate the optimizer ever sees. Print the real peak --
    # the first run diverged (51% -> 58% WER) at a peak of 1.6e-4 that nothing surfaced.
    d_model = train_cfg.get("sched_d_model", 1024)
    peak_lr = lr * d_model ** -0.5 * warmup_steps ** -0.5
    print(f"LR schedule: NoamAnnealing warmup={warmup_steps:,} -> peak {peak_lr:.2e} at step {warmup_steps:,}")
    print(
        "Hebrew: real fine-tune via he-IL langID prompt (slot 64 in base model) — "
        "not a trick; weights adapt to Hebrew audio+text"
    )

    is_tarred = args.tarred or data_cfg["tarred"]["enabled"]
    max_duration = max_clip_duration
    train_ds_overrides = [
        f"model.train_ds.is_tarred={'true' if is_tarred else 'false'}",
        f"model.train_ds.batch_duration={batch_duration}",
        f"model.train_ds.sample_rate={sample_rate}",
        f"model.train_ds.default_prompt_mode=langID",
        "model.train_ds.unified_auto_ratio=0",
        "model.train_ds.shuffle=true",
        f"model.train_ds.max_duration={max_duration}",
    ]
    if num_workers is not None:
        train_ds_overrides.append(f"model.train_ds.num_workers={num_workers}")
        train_ds_overrides.append(f"model.validation_ds.num_workers={min(num_workers, 2)}")
    if is_tarred:
        tar_dir = Path(data_cfg["tarred"]["target_dir"])
        tarred_manifest = tar_dir / "train_tarred.json"
        tar_pattern = str(tar_dir / "audio_*.tar")
        if not tarred_manifest.exists():
            sys.exit(f"Tarred manifest missing: {tarred_manifest}\nRun: python scripts/convert_to_tarred.py")
        train_ds_overrides.extend([
            f"model.train_ds.manifest_filepath={tarred_manifest.resolve()}",
            f"model.train_ds.tarred_audio_filepaths='{tar_pattern}'",
        ])
    else:
        train_ds_overrides.append(f"model.train_ds.manifest_filepath={train_manifest.resolve()}")

    if not args.no_augment:
        augmentor = resolve_augmentor_cfg(cfg)
        if augmentor:
            noise_manifest = Path(cfg["training"]["augmentation"].get("noise_manifest", ""))
            if augmentor.get("noise") and noise_manifest and not noise_manifest.exists():
                print(f"WARNING: noise manifest missing ({noise_manifest})")
                print("  Run: python scripts/build_noise_manifest.py --noise-dir /path/to/noise")
                print("  Continuing without noise augmentation.")
                augmentor = {k: v for k, v in augmentor.items() if k != "noise"}
            elif augmentor.get("noise") and noise_manifest:
                augmentor = dict(augmentor)
                augmentor["noise"] = dict(augmentor["noise"])
                augmentor["noise"]["manifest_path"] = str(noise_manifest.resolve())
            train_ds_overrides.extend(augmentor_hydra_overrides(augmentor))
            print("Train augmentations enabled (telephony preset)")

    val_examples = train_cfg.get("val_log_examples", 2)
    log_every = train_cfg.get("log_every_n_steps", 100)
    wer_log = Path(train_cfg.get("wer_log", args.exp_dir / exp_name / "wer.jsonl"))
    sync_bn = train_cfg.get("sync_batchnorm", devices > 1)
    sched_d_model = train_cfg.get("sched_d_model", 1024)
    optim_betas = train_cfg.get("optim_betas", [0.9, 0.98])
    grad_clip = train_cfg.get("gradient_clip_val")
    parts = [
        f"{sys.executable} {finetune_entry}",
        f"--config-path={cfg_dir}",
        f"--config-name={cfg_name}",
        f"+init_from_nemo_model={base_nemo.resolve()}",
        *train_ds_overrides,
        f"model.validation_ds.manifest_filepath={dev_manifest.resolve()}",
        f"model.validation_ds.sample_rate={sample_rate}",
        f"+model.validation_ds.default_prompt_mode=langID",
        f"model.optim.name=adamw",
        f"model.optim.lr={lr}",
        f"model.optim.weight_decay={train_cfg['weight_decay']}",
        f"model.optim.betas={_hydra_value(optim_betas)}",
        f"model.optim.sched.name=NoamAnnealing",
        f"model.optim.sched.warmup_steps={warmup_steps}",
        f"model.optim.sched.d_model={sched_d_model}",
    ]

    # Cache-aware context. Without this the encoder keeps the checkpoint's own context
    # while eval/inference run at config's att_context_size -- train/serve mismatch.
    att_context = args.att_context or _hydra_value(train_cfg.get("att_context_size"))
    if train_cfg.get("att_context_size") is not None or args.att_context:
        parts.append(f"model.encoder.att_context_size={att_context}")
        print(f"Cache-aware context (train == eval): att_context_size={att_context}")
    if max_steps is not None:
        parts.append(f"trainer.max_steps={max_steps}")
    parts.extend([
        f"trainer.accumulate_grad_batches={grad_accum}",
        f"trainer.devices={devices}",
        f"trainer.num_nodes={num_nodes}",
        f"trainer.accelerator=gpu",
        f"trainer.precision={train_cfg['precision']}",
        f"trainer.val_check_interval={val_every}",
        f"trainer.sync_batchnorm={'true' if sync_bn else 'false'}",
        f"trainer.use_distributed_sampler=false",
        f"exp_manager.exp_dir={args.exp_dir.resolve()}",
        f"exp_manager.name={exp_name}",
        f"exp_manager.checkpoint_callback_params.monitor=val_wer",
        f"exp_manager.checkpoint_callback_params.mode=min",
        f"exp_manager.checkpoint_callback_params.save_top_k={save_top_k}",
        f"+exp_manager.checkpoint_callback_params.save_last=false",
        f"exp_manager.checkpoint_callback_params.always_save_nemo=true",
        f"exp_manager.create_tensorboard_logger=true",
        f"model.log_prediction=true",
        f"trainer.log_every_n_steps={log_every}",
    ])
    if grad_clip is not None:
        parts.append(f"trainer.gradient_clip_val={grad_clip}")

    # Stop on plateau rather than a fixed budget: keep training while val_wer improves.
    early_cfg = train_cfg.get("early_stopping") or {}
    if early_cfg.get("enabled") and not args.no_early_stopping:
        parts.extend([
            "++exp_manager.create_early_stopping_callback=true",
            f"++exp_manager.early_stopping_callback_params.monitor={early_cfg.get('monitor', 'val_wer')}",
            f"++exp_manager.early_stopping_callback_params.mode={early_cfg.get('mode', 'min')}",
            f"++exp_manager.early_stopping_callback_params.patience={early_cfg.get('patience', 5)}",
            f"++exp_manager.early_stopping_callback_params.min_delta={early_cfg.get('min_delta', 0.001)}",
        ])
        print(
            f"Early stopping: monitor={early_cfg.get('monitor', 'val_wer')} "
            f"patience={early_cfg.get('patience', 5)} "
            f"(no max_steps cap — training runs while val_wer improves)"
        )
    if val_every_epochs is not None:
        parts.append(f"+trainer.check_val_every_n_epoch={val_every_epochs}")
    if limit_opt_steps is not None:
        # Lightning counts micro-batches, not optimizer steps.
        parts.append(f"trainer.limit_train_batches={limit_opt_steps * grad_accum}")
    else:
        # Full manifest: override NeMo yaml demo default (1000). This sets epoch length for
        # Lightning val scheduling — not a data cap (~3,447 h still train per epoch).
        parts.append(f"trainer.limit_train_batches={opt_steps_full * grad_accum}")
    limit_val = train_cfg.get("limit_val_batches")
    if limit_val is not None:
        parts.append(f"trainer.limit_val_batches={limit_val}")
    if resume_ckpt is not None:
        parts.append(f"+exp_manager.resume_from_checkpoint={resume_ckpt}")

    cmd = " \\\n  ".join(parts)
    if args.dry_run:
        print(cmd)
        return

    setup_training_env(devices=devices)
    os.environ["NEMO_VAL_LOG_EXAMPLES"] = str(val_examples)
    os.environ["NEMO_FINETUNE_SCRIPT"] = str(finetune_script)
    os.environ["PYTHONPATH"] = str(nemo) + os.pathsep + os.environ.get("PYTHONPATH", "")
    print(f"WER log -> {wer_log.resolve()}  (also: uv run scripts/show_wer.py)")
    run_training(
        cmd.replace("\\\n  ", " "),
        wer_log=wer_log,
        exp_dir=args.exp_dir / exp_name,
        checkpoint_top_k=save_top_k,
        checkpoint_sync_delay_sec=int(ckpt_sync_delay),
    )

    try:
        best_nemo = consolidate_exp_checkpoints(args.exp_dir, exp_name, k=save_top_k)
    except FileNotFoundError:
        best_nemo = find_latest_nemo(args.exp_dir)
    out_link = Path("checkpoints/hebrew-finetuned.nemo")
    out_link.parent.mkdir(parents=True, exist_ok=True)
    if out_link.exists() or out_link.is_symlink():
        out_link.unlink()
    out_link.symlink_to(best_nemo.resolve())
    print(f"\nFine-tuned checkpoint (best val_wer): {best_nemo}")
    print(f"Symlink -> {out_link.resolve()}")
    print("\nEvaluate at deployment latency (blog Step 3):")
    print("  python scripts/eval_streaming.py --compare-base")


if __name__ == "__main__":
    main()
