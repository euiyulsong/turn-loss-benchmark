# CoQA follow-up: genuine history-dependent turns

## Protocol

We prepared the first three consecutive turns from each CoQA story: 1,000 training stories and 100 held-out validation stories (3,000 answer targets for train; 300 for evaluation). Unlike the SQuAD grouping pilot, CoQA follow-up questions can depend on prior turns (for example, “Whose future is uncertain?” → “Where exactly?” → “Why?”).

We trained two assistant-only LoRA conditions from the same base model, seed 42, batch size 1, one epoch, and 1,000 optimizer updates:

- **single:** each answer target is a separate sequence. For turns 2–3, the passage and gold prior questions/answers are flattened into the current user message.
- **multi:** each story is one sequence, with earlier questions and gold answers as separate user/assistant turns.

Thus, the comparison preserves history content and answer targets; it changes the representation of prior conversation. This is not a comparison against history-free standalone questions.

Evaluation reports three views for each turn: `single` with flattened gold history, `multi_gold` with the same gold history represented in separate turns, and `multi_rollout` where the model's own prior answers are fed back. Rollout reflects both current-turn format and error propagation. Scores are macro averages over 300 answers, single-reference SQuAD-style EM/F1; they are not the official CoQA evaluator scores.

## Adapter evaluation (complete)

| Trained condition | Evaluation view | EM | F1 |
|---|---|---:|---:|
| Single | Flattened history (`single`) | 0.610 | 0.7521 |
| Single | Gold multi-turn history (`multi_gold`) | 0.600 | 0.7483 |
| Single | Model-answer rollout (`multi_rollout`) | 0.587 | 0.7311 |
| Multi | Flattened history (`single`) | 0.600 | 0.7460 |
| Multi | Gold multi-turn history (`multi_gold`) | 0.593 | 0.7471 |
| Multi | Model-answer rollout (`multi_rollout`) | 0.597 | 0.7456 |

On this pilot, the multi-trained model is essentially tied with the single-trained model on the gold-history multi-turn F1 (0.7471 vs. 0.7483), while its rollout F1 is higher (0.7456 vs. 0.7311). The apparent rollout difference is small and should not be called a reliable multi-turn advantage without paired uncertainty and additional training seeds.

## Paired passage-bootstrap comparisons

Differences are multi-trained minus single-trained F1 or fine-tuned minus base F1, with 95% cluster-bootstrap intervals over the 100 held-out stories (5,000 resamples). The intervals quantify story-sampling uncertainty for this one trained seed, not training-seed variability.

### Multi-trained vs. single-trained

| Evaluation view | F1 difference | 95% CI |
|---|---:|---:|
| Flattened gold history (`single`) | -0.0061 | [-0.0265, 0.0142] |
| Gold multi-turn history (`multi_gold`) | -0.0011 | [-0.0205, 0.0165] |
| Model-answer rollout (`multi_rollout`) | +0.0144 | [-0.0081, 0.0361] |

All three intervals include zero. This pilot therefore does **not** show a reliable advantage for multi-format training over the history-preserving single-format control, including on rollout.

### Fine-tuned vs. base model

| Fine-tuned condition | Evaluation view | F1 difference | 95% CI |
|---|---|---:|---:|
| Multi | Flattened gold history (`single`) | +0.0756 | [0.0426, 0.1096] |
| Multi | Gold multi-turn history (`multi_gold`) | +0.0813 | [0.0499, 0.1142] |
| Multi | Model-answer rollout (`multi_rollout`) | +0.0963 | [0.0604, 0.1341] |
| Single | Flattened gold history (`single`) | +0.0817 | [0.0467, 0.1175] |
| Single | Gold multi-turn history (`multi_gold`) | +0.0824 | [0.0514, 0.1144] |
| Single | Model-answer rollout (`multi_rollout`) | +0.0819 | [0.0459, 0.1186] |

Both fine-tuned conditions improve over base on these held-out stories. The comparison between training formats remains inconclusive; the multi format's rollout point estimate is higher, but its paired interval includes zero.

## Training accounting

| Condition | Input tokens / epoch | Assistant loss tokens / epoch | Optimizer steps | Train time |
|---|---:|---:|---:|---:|
| Single / assistant-only | 1,262,732 | 22,935 | 1,000 | 425.0 s |
| Multi / assistant-only | 448,119 | 22,935 | 1,000 | 148.6 s |

The target and supervised assistant-token counts match, but the single format repeats the passage and therefore processes substantially more input tokens. The comparison is not compute-matched.

## Interpretation limits

- One seed and one epoch; these scores alone do not establish which format generalizes better.
- The 100-story passage/story bootstrap captures evaluation-sample uncertainty, not training-seed uncertainty.
- `multi_gold` is the cleanest view of whether the explicit multi-turn role structure helps when prior correct answers are available. `multi_rollout` is closer to deployment but compounds prior answer errors.
- CoQA uses multiple references/annotators in its official evaluation. Here each question uses the dataset's `input_text` answer only and a SQuAD-style normalizer, so values are an internal matched comparison, not leaderboard-compatible CoQA scores.

Prediction JSONL and model adapters remain local and are not committed. The evaluation outputs include 900 rows per adapter (100 stories × 3 turns × 3 views).
