# SQuAD pilot results

## Summary

We ran a 2×2 LoRA fine-tuning comparison on 1,000 grouped SQuAD passages and evaluated on 100 held-out passages (300 QA targets per evaluation mode). All four training runs used Qwen/Qwen2.5-1.5B-Instruct, seed 42, batch size 1, one epoch, and 1,000 optimizer steps. The base checkpoint was evaluated on the same targets.

The highest scores came from multi-format, assistant-only training. Assistant-only training also outperformed full prompt-and-answer loss in both formats. These are pilot results from one seed and should not be treated as a multi-seed conclusion.

## Results

Values are EM / token-normalized word-overlap F1. Each mode has 300 QA targets.

| Checkpoint | Single QA | Multi gold-context | Multi rollout |
|---|---:|---:|---:|
| Base | 0.787 / 0.877 | 0.807 / 0.898 | 0.803 / 0.901 |
| Single / assistant-only | 0.837 / 0.914 | 0.850 / 0.922 | 0.843 / 0.919 |
| Single / full loss | 0.760 / 0.859 | 0.753 / 0.860 | 0.753 / 0.856 |
| Multi / assistant-only | **0.883 / 0.940** | **0.883 / 0.938** | **0.887 / 0.939** |
| Multi / full loss | 0.810 / 0.899 | 0.810 / 0.898 | 0.813 / 0.905 |

## Paired passage bootstrap comparisons

The following are mean F1 differences (left minus right), with 95% cluster-bootstrap confidence intervals over 100 passages (5,000 resamples). They quantify passage-sampling uncertainty for this trained seed, not training-seed uncertainty.

### Multi vs. single, assistant-only

| Evaluation | F1 difference | 95% CI |
|---|---:|---:|
| Single QA | +0.0262 | [0.0039, 0.0501] |
| Multi gold-context | +0.0157 | [-0.0003, 0.0322] |
| Multi rollout | +0.0201 | [0.0045, 0.0363] |

The observed gain is positive in all modes; this interval excludes zero for single QA and rollout, but marginally includes zero for gold-context evaluation.

### Assistant-only vs. full loss

| Format | Evaluation | F1 difference | 95% CI |
|---|---|---:|---:|
| Single | Single QA | +0.0549 | [0.0224, 0.0900] |
| Single | Multi gold-context | +0.0629 | [0.0304, 0.0957] |
| Single | Multi rollout | +0.0631 | [0.0289, 0.0978] |
| Multi | Single QA | +0.0412 | [0.0186, 0.0656] |
| Multi | Multi gold-context | +0.0403 | [0.0169, 0.0649] |
| Multi | Multi rollout | +0.0345 | [0.0061, 0.0626] |

Assistant-only is higher in all six comparisons, and the passage-bootstrap intervals exclude zero. Full loss changes the relative weighting of assistant tokens as well as adding prompt-token loss, so this is not an isolated causal estimate of adding prompt loss.

### Fine-tuned models vs. base

| Fine-tuned checkpoint | Evaluation | F1 difference | 95% CI |
|---|---|---:|---:|
| Multi / assistant-only | Single QA | +0.0628 | [0.0352, 0.0919] |
| Multi / assistant-only | Multi gold-context | +0.0400 | [0.0103, 0.0693] |
| Multi / assistant-only | Multi rollout | +0.0379 | [0.0062, 0.0686] |
| Single / assistant-only | Single QA | +0.0366 | [0.0048, 0.0690] |
| Single / assistant-only | Multi gold-context | +0.0243 | [-0.0066, 0.0536] |
| Single / assistant-only | Multi rollout | +0.0178 | [-0.0140, 0.0483] |
| Multi / full loss | Single QA | +0.0216 | [-0.0076, 0.0503] |
| Multi / full loss | Multi gold-context | -0.0003 | [-0.0284, 0.0279] |
| Multi / full loss | Multi rollout | +0.0034 | [-0.0271, 0.0332] |
| Single / full loss | Single QA | -0.0184 | [-0.0476, 0.0111] |
| Single / full loss | Multi gold-context | -0.0387 | [-0.0725, -0.0059] |
| Single / full loss | Multi rollout | -0.0453 | [-0.0802, -0.0110] |

## Training details

| Format / objective | Train targets | Input tokens / epoch | Loss tokens / epoch | Optimizer steps | Train time |
|---|---:|---:|---:|---:|---:|
| Single / assistant-only | 3,000 | 646,835 | 27,805 | 1,000 | 503.3 s |
| Single / full loss | 3,000 | 646,835 | 646,835 | 1,000 | 503.2 s |
| Multi / assistant-only | 3,000 | 275,481 | 27,805 | 1,000 | 176.1 s |
| Multi / full loss | 3,000 | 275,481 | 275,481 | 1,000 | 148.8 s |

Batch size was 1 in all conditions. Single-format training used gradient accumulation of 3; multi used 1. Although target counts and optimizer steps match, single-format inputs repeat the passage and therefore use more input tokens. The single and multi objectives also have different token-length weighting under ordinary token-mean cross entropy.

## Protocol and limitations

- SQuAD 1.1 passages with at least three independent questions were grouped into three-turn examples. Train/eval passages were disjoint.
- Evaluation used deterministic greedy decoding and 64 maximum new tokens. `single` uses no prior turns; `multi_gold` supplies gold prior answers; `multi_rollout` supplies the model's prior answers.
- EM and SQuAD-style word-overlap F1 are reported. This is a 100-passage pilot, one training seed, and one epoch. Bootstrap intervals do not include seed variation.
- SQuAD is originally single-turn QA; grouping independent questions does not test natural history-dependent dialogue.
- Full loss is the ordinary token-mean CE over prompt and assistant spans. It changes both prompt supervision and the relative weight of assistant spans. A separately normalized objective and lambda sweep would be needed to isolate prompt loss.
- Prepared source data, full prediction JSONL files, and LoRA adapter weights are intentionally not committed. See the SQuAD source and license links in [README.md](README.md).

## Reproduction

Install `requirements.txt`, then prepare 1,000 train groups and 100 held-out groups and run the 1-epoch four-condition experiment:

```bash
EPOCHS=1 SEEDS="42" TRAIN_GROUPS=1000 EVAL_GROUPS=100 bash run.sh
```

The experiment runner evaluates the base checkpoint and each adapter, then writes paired comparison JSON files. GPU runtimes vary by hardware and load.
