# Setup

Create a new environment using conda.

```bash
conda create -p .venv_lflexmdm python=3.11.10 pip ipykernel -y
conda activate ./.venv_lflexmdm
pip install -r requirements.txt
```

Create `.env` file in the root directory from where you plan to run the experiments:

```bash
# wandb
WANDB_ENTITY=???
WANDB_PROJECT=???
# paths to appropriate directories
DATA_DIR=data
HF_HOME=hf_home
HF_DATASETS_CACHE=hf_datasets_cache
# output
LOG_DIR=logs
# misc
TOKENIZERS_PARALLELISM=false
PROJECT_ROOT=.
# hydra
HYDRA_FULL_ERROR=1
OC_CAUSE=1
# emails from slurm scheduler if applicable
EMAIL=???
# torch compile logs
TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
# Hugging Face Hub (for ++hub.repo_id eval)
# HF_TOKEN=...  or HF_HUB_KEY=...
```

# Pretrained weights (Hugging Face Hub)

Bracket-SAFE molgen checkpoints (EMA, step 50k). Paper: [Variable-Length Discrete Diffusion](https://arxiv.org/pdf/2602.18695).

| Model                           | Hub repo                                                                                                        | Hydra experiment                      |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------|---------------------------------------|
| FlexMDM (medium)                | [FlexMDM-Medium-BracketSAFE](https://huggingface.co/dhruveshpatel/FlexMDM-Medium-BracketSAFE)                   | `safe_flexmdm`                        |
| LoFlexMDM medium (12-layer aux) | [LoFlexMDM-Medium-BracketSAFE](https://huggingface.co/dhruveshpatel/LoFlexMDM-Medium-BracketSAFE)               | `[safe_lflexmdm,b_um_1_a1]`           |
| LoFlexMDM tiny (8-layer aux)    | [LoFlexMDM-Tiny-BracketSAFE](https://huggingface.co/dhruveshpatel/LoFlexMDM-Tiny-BracketSAFE)                   | `[safe_lflexmdm,b_um_1_a1,tiny_aux]`  |
| LoFlexMDM xtiny (6-layer aux)   | [LoFlexMDM-xTiny-BracketSAFE](https://huggingface.co/dhruveshpatel/LoFlexMDM-xTiny-BracketSAFE)                 | `[safe_lflexmdm,b_um_1_a1,xtiny_aux]` |
| LoFlexMDM shared backbone       | [LoFlexMDM-Medium-Shared-BracketSAFE](https://huggingface.co/dhruveshpatel/LoFlexMDM-Medium-Shared-BracketSAFE) | `[safe_lflexmdm_shared,b_um_1_a1]`    |

Eval loads weights with `++hub.repo_id=...` (downloads `model.safetensors` from Hub). STAR graph-traversal checkpoints are not on Hub; use local Lightning checkpoints if needed.

# Eval

## Paper best hyperparameters (Bracket-SAFE molgen)

From the paper Table 1 grid (see also Hub model cards):

- `predictor.max_steps=1024`
- `predictor.confidence=top_prob` (use `null` for no confidence decoding)
- `predictor.top_p=0.5` (best quality; paper also reports `0.9` and `1.0`)
- LFlexMDM runs use `b_um_1_a1` (fixed unmask rate)


## De novo molecule generation

Common flags for eval:
```bash
COMMON_FLAGS="++skip_init_weights=true ++eval.split=test ~datamodule.dataset_managers.train ++eval.split=test per_device_batch_size=64 per_device_val_batch_size=64 global_batch_size=64 trainer_strategy=single_device ++trainer.precision=32-true compile=false"
```

**(Baseline) FlexMDM on molecule generation**

```bash
DATASET=safe
CONFIDENCE=null # paper best; or null
TOP_P=0.5           # paper best; or 0.9 or 1.0
HUB_REPO=dhruveshpatel/FlexMDM-Medium-BracketSAFE
for SEED in 42 231 10234; do
    for TOP_P in 0.5 0.9 1.0; do
    for MAX_STEPS in 1024 512 256 128; do
        for CONFIDENCE in top_prob null; do
            xlm job_type=eval job_name=${DATASET}_flexmdm_${SEED}_${TOP_P}_${CONFIDENCE}_${MAX_STEPS} experiment=safe_flexmdm \
            ++hub.repo_id=${HUB_REPO} ${COMMON_FLAGS} \
            datamodule.dataset_managers.test.unconditional_prediction.num_examples=1000 \
            ++predictor.confidence=${CONFIDENCE} predictor.max_steps=${MAX_STEPS} \
            ++predictor.top_k=null ++predictor.top_p=${TOP_P} ++seed=${SEED}
        done
    done
    done
done
```

**(Ours) LFlexMDM on molecule generation**

```bash
DATASET=safe
CONFIDENCE=top_prob # paper best; or null
TOP_P=0.5           # paper best; or 0.9 or 1.0
BACKBONE=separate   # or shared
aux_size=medium     # or tiny or xtiny (separate backbone only)
if [ "$BACKBONE" = "separate" ]; then
    if [ "$aux_size" = "medium" ]; then
        EXPERIMENT="[${DATASET}_lflexmdm,b_um_1_a1]"
        HUB_REPO=dhruveshpatel/LoFlexMDM-Medium-BracketSAFE
    elif [ "$aux_size" = "tiny" ]; then
        EXPERIMENT="[${DATASET}_lflexmdm,b_um_1_a1,tiny_aux]"
        HUB_REPO=dhruveshpatel/LoFlexMDM-Tiny-BracketSAFE
    else
        EXPERIMENT="[${DATASET}_lflexmdm,b_um_1_a1,xtiny_aux]"
        HUB_REPO=dhruveshpatel/LoFlexMDM-xTiny-BracketSAFE
    fi
else
    EXPERIMENT="[${DATASET}_lflexmdm_shared,b_um_1_a1]"
    HUB_REPO=dhruveshpatel/LoFlexMDM-Medium-Shared-BracketSAFE
fi
for SEED in 42 231 10234; do
    for TOP_P in 0.5 0.9 1.0; do
        for CONFIDENCE in top_prob null; do
            for MAX_STEPS in 1024 512 256 128; do
                xlm job_type=eval job_name=${DATASET}_lflexmdm_${BACKBONE}_${aux_size}_${SEED}_${TOP_P}_${CONFIDENCE}_${MAX_STEPS} experiment=${EXPERIMENT} \
                ++hub.repo_id=${HUB_REPO} ${COMMON_FLAGS} \
                datamodule.dataset_managers.test.unconditional_prediction.num_examples=1000 \
                ++predictor.confidence=${CONFIDENCE} predictor.max_steps=${MAX_STEPS} \
                ++predictor.top_k=null ++predictor.top_p=${TOP_P} ++seed=${SEED}
            done
        done
    done
done
```

## Fragment  constrained generation

Requires `data/genmol_fragments.csv` under `DATA_DIR`.


**(Baseline) FlexMDM fragment eval**

```bash
for TASK in motif_extension scaffold_decoration superstructure_generation linker_design; do
    for SEED in 42 231 10234; do
        for TOP_P in 0.5 0.9 1.0; do
            for CONFIDENCE in top_prob null; do
                for MAX_STEPS in 1024 512 256 128; do
                xlm job_type=eval job_name=safe_flexmdm_eval_${TASK} \
                experiment=[safe_flexmdm,safe_conditional_eval] \
                ++hub.repo_id=dhruveshpatel/FlexMDM-Medium-BracketSAFE \
                ${COMMON_FLAGS} \
                molgen_task_name=${TASK} \
                num_conditional_prediction_samples=100 \
                ++predictor.confidence=${CONFIDENCE} \
                predictor.max_steps=${MAX_STEPS} \
                ++predictor.top_k=null ++predictor.top_p=${TOP_P} ++seed=${SEED}
            done
        done
    done
done
```

**(Ours) LFlexMDM fragment eval**

```bash
BACKBONE=separate   # or shared
aux_size=medium     # or tiny or xtiny (separate only)
if [ "$BACKBONE" = "separate" ]; then
    if [ "$aux_size" = "medium" ]; then
        EXPERIMENT="[safe_lflexmdm,b_um_1_a1,safe_conditional_eval]"
        HUB_REPO=dhruveshpatel/LoFlexMDM-Medium-BracketSAFE
    elif [ "$aux_size" = "tiny" ]; then
        EXPERIMENT="[safe_lflexmdm,b_um_1_a1,tiny_aux,safe_conditional_eval]"
        HUB_REPO=dhruveshpatel/LoFlexMDM-Tiny-BracketSAFE
    else
        EXPERIMENT="[safe_lflexmdm,b_um_1_a1,xtiny_aux,safe_conditional_eval]"
        HUB_REPO=dhruveshpatel/LoFlexMDM-xTiny-BracketSAFE
    fi
else
    EXPERIMENT="[safe_lflexmdm_shared,b_um_1_a1,safe_conditional_eval]"
    HUB_REPO=dhruveshpatel/LoFlexMDM-Medium-Shared-BracketSAFE
fi
for TASK in motif_extension scaffold_decoration superstructure_generation linker_design; do
    for SEED in 42 231 10234; do
        for TOP_P in 0.5 0.9 1.0; do
            for CONFIDENCE in top_prob null; do
                for MAX_STEPS in 1024 512 256 128; do
                xlm job_type=eval job_name=safe_lflexmdm_eval_${BACKBONE}_${aux_size}_${TASK} \
                experiment=${EXPERIMENT} ++hub.repo_id=${HUB_REPO} \
                ${COMMON_FLAGS} \
                molgen_task_name=${TASK} \
                num_conditional_prediction_samples=100 \
                ++predictor.confidence=${CONFIDENCE} \
                predictor.max_steps=${MAX_STEPS} \
                ++predictor.top_k=null ++predictor.top_p=${TOP_P} ++seed=${SEED}
            done
        done
    done
done
```


# Train

**(Baseline) FlexMDM on Graph Traversal**

```bash
DATASET=star_hard # or star_easy, star_medium
MODEL=flexmdm
EXPERIMENT=${DATASET}_${MODEL}
xlm job_name=${DATASET}_${MODEL} job_type=train experiment=${DATASET}_${MODEL} per_device_batch_size=64 trainer_strategy=single_device trainer.devices=1 trainer.num_nodes=1 ++trainer.precision=bf16-mixed compile=False
```

**(Ours) LoFlexMDM on Graph Traversal**

```bash
DATASET=star_hard # or star_easy, star_medium
MODEL=lflexmdm
BACKBONE=separate # or shared
aux_size=medium # or tiny or xtiny
if [ "$BACKBONE" = "separate" ]; then
    if [ "$aux_size" = "medium" ]; then
        EXPERIMENT="[${DATASET}_${MODEL},b_um_1_a1]"
    else
        EXPERIMENT="[${DATASET}_${MODEL},b_um_1_a1,${aux_size}_aux]"
    fi
else
    EXPERIMENT="[${DATASET}_${MODEL}_shared,b_um_1_a1]"
fi
xlm job_name=${DATASET}_${MODEL}_${BACKBONE} job_type=train experiment=${EXPERIMENT} per_device_batch_size=64 trainer_strategy=single_device trainer.devices=1 trainer.num_nodes=1 ++trainer.precision=bf16-mixed compile=False
```

**(Ours) LoFlexMDM on SAFE molecules**

```bash
DATASET=safe
MODEL=lflexmdm
BACKBONE=separate # or shared 
aux_size=medium # or tiny or xtiny
if [ "$BACKBONE" = "separate" ]; then
    EXPERIMENT="[${DATASET}_${MODEL},b_um_1_a1,${aux_size}_aux]"
else
    EXPERIMENT="[${DATASET}_${MODEL}_shared,b_um_1_a1]"
fi
xlm \
    job_name=${DATASET}_${MODEL}_${BACKBONE}_${aux_size} \
    job_type=train experiment=${EXPERIMENT} \
    per_device_batch_size=128 trainer_strategy=ddp trainer.devices=8 trainer.num_nodes=1 \
    ++trainer.precision=bf16-mixed compile=true \
    global_batch_size=2048
```

# Acknowledgments
The code uses [xlm-core](https://github.com/dhruvdcoder/xlm-core) as the rapid experiment framework and builds on [FlexMDM](https://github.com/brianlck/FlexMDM). The code for data pipeline for the graph traversal experiments is from [ILM](https://github.com/dhruvdcoder/ILM) and for the molecule generation experiments is adapted from [GenMol](https://github.com/NVIDIA-Digital-Bio/genmol).

# Citation
Please cite the following for the method LoFlexMDM:
```
@misc{patel2026insertionbasedsequencegeneration,
      title={Insertion Based Sequence Generation with Learnable Order Dynamics}, 
      author={Dhruvesh Patel and Benjamin Rozonoyer and Gaurav Pandey and Tahira Naseem and Ramón Fernandez Astudillo and Andrew McCallum},
      year={2026},
      eprint={2602.18695},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.18695}, 
}
```
and cite xlm-core for the framework:

```
@article{patel2025xlm,
  title={XLM: A Python package for non-autoregressive language models},
  author={Patel, Dhruvesh and Maram, Durga Prasad and Chintha, Sai Sreenivas and Rozonoyer, Benjamin and McCallum, Andrew},
  journal={arXiv preprint arXiv:2512.17065},
  year={2025}
}
```
