#!/bin/bash
# Overfit sanity test, fast harness: no .nemo saves (2.5 GB each was 99% of wall time),
# validate every 50 epochs (100 steps) instead of every epoch. Same 60-clip set, same
# schedule as the interrupted run (lr 0.1 / warmup 200 -> peak 2.2e-4).
set -u
export NEMO_FINETUNE_SCRIPT=/root/NeMo-main/examples/asr/speech_to_text_finetune.py
export PYTHONPATH=/root/NeMo-main:/root/stubs
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd /root/nemo-he

exec /opt/venv/bin/python /root/nemo-he/scripts/nemo_finetune_entry.py \
  --config-path=/root/NeMo-main/examples/asr/conf/fastconformer/cache_aware_streaming \
  --config-name=fastconformer_transducer_bpe_streaming_prompt \
  +init_from_nemo_model=/root/nemo-he/checkpoints/nemotron-3.5-asr-base.nemo \
  model.train_ds.is_tarred=false \
  model.train_ds.batch_duration=100 \
  model.train_ds.sample_rate=16000 \
  model.train_ds.default_prompt_mode=langID \
  model.train_ds.unified_auto_ratio=0 \
  model.train_ds.shuffle=true \
  model.train_ds.max_duration=40 \
  model.train_ds.num_workers=4 \
  model.train_ds.manifest_filepath=/root/data/manifests/overfit.json \
  model.validation_ds.manifest_filepath=/root/data/manifests/overfit.json \
  model.validation_ds.sample_rate=16000 \
  model.validation_ds.num_workers=2 \
  model.validation_ds.batch_duration=200 \
  +model.validation_ds.default_prompt_mode=langID \
  model.optim.name=adamw \
  model.optim.lr=0.1 \
  model.optim.weight_decay=0.001 \
  model.optim.betas=[0.9,0.98] \
  model.optim.sched.name=NoamAnnealing \
  model.optim.sched.warmup_steps=200 \
  model.optim.sched.d_model=1024 \
  model.encoder.att_context_size=[56,13] \
  trainer.max_steps=500 \
  trainer.accumulate_grad_batches=1 \
  trainer.devices=2 \
  trainer.num_nodes=1 \
  trainer.accelerator=gpu \
  trainer.precision=bf16 \
  trainer.val_check_interval=1.0 \
  ++trainer.check_val_every_n_epoch=50 \
  trainer.sync_batchnorm=true \
  trainer.use_distributed_sampler=false \
  trainer.limit_train_batches=2 \
  trainer.log_every_n_steps=100 \
  trainer.gradient_clip_val=0.5 \
  exp_manager.exp_dir=/root/exp_overfit \
  exp_manager.name=hebrew_ft \
  exp_manager.checkpoint_callback_params.monitor=val_wer \
  exp_manager.checkpoint_callback_params.mode=min \
  exp_manager.checkpoint_callback_params.save_top_k=1 \
  +exp_manager.checkpoint_callback_params.save_last=false \
  exp_manager.checkpoint_callback_params.always_save_nemo=false \
  exp_manager.create_tensorboard_logger=false \
  model.log_prediction=true
