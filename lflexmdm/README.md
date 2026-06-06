# Star
```bash
python scripts/submit_nll_eval.py "do=submit" "job_name=star_easy_flexmdm_eval" "eval.num_examples=null" "eval.experiment=star_easy_flexmdm" "eval.generative_perplexity=null" "eval.eval_type=exact_match" "slurm.constrain=vram40" "slurm.partition=\"gpu,gpu-preempt\"" "eval.checkpoint_path=/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/xlm/logs/star_easy_flexmdm_v2/checkpoints/102-80000.ckpt"  use_job_name_as_id=false --- ++predictor.confidence=null predictor.max_steps=200
```

# LM1B

```bash
python scripts/submit_train.py "do=submit" "job_name=lm1b_flexmdm" "train.experiment=lm1b_flexmdm" "train.batch_size=128" "train.compile=false" "train.precision=bf16-mixed" "hardware=ddp_4_node_1_gpu" "slurm.constraint=\"vram80,bf16,ib\"" "++slurm.exclude=gpu016"
```