# Prompt loss × single/multi-turn SFT benchmark

실험 스크립트 및 실행 호환성은 검증 중입니다. 기본 모델은 Qwen2.5-1.5B-Instruct, LoRA입니다. Qwen3 thinking/template 변형은 기본 검증 범위 밖입니다. 설치한 라이브러리 버전이 requirements.txt와 다르면 호환성 이슈가 있을 수 있습니다. SQuAD 파일럿 결과는 [실험 보고서](REPORT.md), 자연스러운 multi-turn CoQA 후속 실험은 [CoQA 보고서](COQA_REPORT.md)에 기록했습니다.

## 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run.sh
# 재현성: 같은 prepared data에 학습 seed만 3개
SEEDS="42 43 44" bash run.sh
# 1 epoch 실행
EPOCHS=1 bash run.sh
# 자연 대화 보조 실험
DATASET=coqa SEEDS="42 43 44" bash run.sh
```

모델/데이터 다운로드에는 Hugging Face 접속이 필요합니다. CPU도 코드상 지원하지만 전체 학습에는 GPU를 권장합니다. VRAM/시간은 실측 전 확정하지 않습니다. 작은 실행은 TRAIN_GROUPS=10 EVAL_GROUPS=2 bash run.sh 입니다. 모델 변경 시 MODEL 환경변수를 지정하고 별도 출력 디렉터리를 사용하세요. run.sh는 재실행 시 같은 이름 결과를 교체합니다. 학습은 단일 GPU용입니다.

## 질문 1: prompt loss가 도움이 되는가?

2×2: single/assistant, single/full, multi/assistant, multi/full. 동일 checkpoint에서 각각 독립 LoRA로 학습합니다. assistant는 assistant message span(role header, 본문, EOT)을 학습합니다. full은 system/user/assistant 모두 학습합니다. 이 데이터에는 tool 메시지가 없습니다. 따라서 tool observation loss 효과까지 검증했다고 해석하면 안 됩니다.

기본값은 일반적인 token-mean cross entropy입니다. full은 prompt loss를 추가하면서 assistant loss의 상대 비중도 줄입니다. 따라서 결과는 '실제 두 training objective의 비교'이고, prompt loss만 추가한 순수 인과 효과로 해석하지 마세요. 순수 효과를 추가 검증하려면 L=mean(assistant CE)+lambda*mean(prompt CE) 구현으로 lambda sweep을 별도로 해야 합니다.

## 질문 2: multi-only 학습이 single-only보다 single-turn에서 좋은가?

### 주 실험: SQuAD 1.1 — 독립적인 passage QA

같은 context에 독립 question이 3개 이상 있는 passage를 선택합니다. 동일 3,000개 QA target을 두 형태로 만듭니다.

single: 매번 system + passage/question + answer, 3,000 sequences.
multi: 첫 user에 passage/question, 다음 2 user에 추가 question, 모든 assistant에 답변, 1,000 sequences.

도메인, task, passage, question, answer, target 노출 횟수를 맞춥니다. 원래 공개 SQuAD는 single-turn 데이터입니다. multi 버전은 같은 passage의 QA를 묶어 생성하며 자연 대화의 대명사/생략은 포함하지 않습니다. 모델이 만든 question/answer나 rewrite는 사용하지 않습니다.

같은 QA만 학습하므로 multi에서 target 수가 늘어나 얻는 이득을 방지합니다. epoch 수는 `EPOCHS`로 설정하고, batch size=1, single grad accumulation=3, multi=1로 optimizer update 수를 맞춥니다. 학습용 target 수가 3배라고 비교하는 오류를 피합니다. 단, single은 passage가 반복되므로 전체 input token 수/FLOPs는 같지 않습니다. run.json에서 input/loss token과 elapsed seconds를 보고합니다. CE 정규화 때문에 답변별 길이 가중치도 완전히 같지는 않습니다. Multi 첫 assistant는 causal mask상 single context로 학습하므로 'multi-only'는 sequence가 multi라는 뜻이지 short-prefix 감독이 전혀 없다는 뜻은 아닙니다.

### 보조 실험: CoQA — 자연스러운 history-dependent QA

CoQA의 첫 3턴을 사용합니다. multi는 원래 대화 구조. single은 매 target에 passage+이전 gold Q/A를 한 user 메시지에 펼칩니다. 따라서 같은 정보와 target을 보존하고 role 구조의 효과를 비교합니다. 이 single은 '대화 role이 한 턴'인 것이며 history-free standalone question이 아닙니다. 후속 질문에서 history를 삭제하면 문제의 의미가 바뀌므로 그러한 비교는 하지 않습니다. CoQA 첫 턴만의 성능도 predictions의 turn=1로 따로 집계할 수 있습니다. SQuAD와 CoQA 간 점수를 직접 비교하지 마세요.

## 평가

공식 validation에서 held-out 100 passages × 3 questions = 300 targets. Train과 eval에 동일 context가 없도록 제거합니다. 같은 passage의 questions가 split을 넘나들지 않습니다. Tokenizer 길이를 모든 조건에서 사전 필터링하고 자르지 않습니다. 길이 필터로 선택 편향이 있으므로 전체 데이터 성능이라고 보고하지 마세요.

모든 checkpoint를 동일 targets에 평가:
- single: SQuAD는 history 없는 genuine single QA; CoQA는 flattened gold history.
- multi_gold: 이전 정답을 주는 teacher-forced multi-turn 평가.
- multi_rollout: 이전 모델 답변을 이어가는 실제 multi-turn 평가.

EM/word-overlap F1, deterministic greedy generation, base checkpoint 포함. CoQA 점수는 단일 reference의 SQuAD-style EM/F1이며 공식 CoQA 다중 annotator evaluator와 동일하지 않습니다. 출력은 short answer만 요구합니다. 생성된 장황한 응답을 임의로 정답에 맞춰 후처리하지 않습니다.

핵심 판단은 results/*turn_effect*.json의 single 필드: multi_assistant − single_assistant F1. passage 단위 paired bootstrap 95% CI가 0을 포함하면 이 규모에서 우위 불확실. 3 seeds의 차이 방향과 크기를 모두 함께 보고하세요. Bootstrap은 한 학습 seed 내부 passage sampling uncertainty이며 seed uncertainty까지 포괄하지 않습니다. 100 passages는 파일럿입니다. 학습 중 checkpoint 선택에 이 eval을 쓰지 않습니다. Hyperparameter 탐색을 한다면 별도 dev split과 최종 test가 필요합니다.

## Mask 검증

커스텀 mask는 chat template의 각 prefix를 비교하여 assistant 구간만 labels에 남깁니다. 이전 prefix가 바뀌는 template는 즉시 실패합니다. assistant 역할 header/EOT 포함 여부는 이 키트에서는 포함으로 고정합니다. Pad labels는 -100. Causal LM의 label shift는 Transformers 모델이 처리합니다. 모든 multi assistant 턴에 loss를 겁니다. Template 없는 단순 문자열 concat이 아닙니다. 순서대로 턴마다 optimizer update하는 방식도 아닙니다.

## 출처 및 라이선스

- SQuAD: https://huggingface.co/datasets/rajpurkar/squad , https://rajpurkar.github.io/SQuAD-explorer/ — CC BY-SA 4.0. 데이터 재배포 시 원 출처와 라이선스 보존.
- CoQA: https://stanfordnlp.github.io/coqa/ , https://huggingface.co/datasets/stanfordnlp/coqa . Source corpus별 조건이 있어 프로젝트 공식 license를 확인하세요. 원 데이터는 키트에 재배포하지 않습니다.
- Llama 3 SFT prompt masking: https://arxiv.org/html/2407.21783v3 §4.1.3.
- TRL assistant-only reference: https://huggingface.co/docs/trl/sft_trainer . 여기서는 TRL 버전 의존 대신 Transformers labels를 직접 구성합니다.

실행 시 `EPOCHS=1`로 epoch 수를 지정할 수 있습니다. per-device batch size는 1입니다.

로컬 검증: Python compile, bash syntax, offline construction/masking/metric smoke tests. GPU 학습/평가 결과는 완료된 출력 파일이 있을 때만 보고합니다.

## CoQA follow-up experiment

The completed CoQA follow-up results, paired confidence intervals, and interpretation are in [COQA_REPORT.md](COQA_REPORT.md). It uses three consecutive human QA turns per story and compares flattened single-turn history against separate conversational turns.

## 추가 파일럿: 0.5B prompt-mask ablation (부분 실행)

별도 실험 폴더에서 `Qwen/Qwen2.5-0.5B-Instruct`로 수행한 1 epoch 결과입니다. 계획한 24개 조합 중 12개가 학습되었고, 그중 **11개만 평가까지 완료**되었습니다. `adv / single / all_loss`는 학습 adapter만 생성되고 평가되지 않아 아래 표와 비교에서 제외했습니다. Seed 42, batch size 1, 데이터셋별 1,000 train answer targets 및 100 validation targets입니다. 아래 값은 SQuAD-style exact match (EM)이며, 각 평가 모드의 n=100입니다.

### SQuAD validation

| 학습 데이터 / 형식 / loss | Single | Multi gold history | Multi rollout |
|---|---:|---:|---:|
| Base (no fine-tuning) | 0.58 | 0.53 | 0.56 |
| SQuAD / single / assistant-only | 0.72 | 0.64 | 0.65 |
| SQuAD / single / system loss | 0.71 | 0.63 | 0.66 |
| SQuAD / single / user loss | 0.63 | 0.55 | 0.57 |
| SQuAD / single / all loss | 0.63 | 0.54 | 0.53 |
| SQuAD / multi / assistant-only | **0.74** | 0.69 | 0.69 |
| SQuAD / multi / system loss | 0.73 | **0.72** | **0.74** |
| SQuAD / multi / user loss | 0.62 | 0.65 | 0.68 |
| SQuAD / multi / all loss | 0.61 | 0.66 | 0.67 |
| AdversarialQA / single / assistant-only | 0.60 | 0.63 | 0.65 |
| AdversarialQA / single / system loss | 0.58 | 0.63 | 0.61 |
| AdversarialQA / single / user loss | 0.56 | 0.54 | 0.48 |

### AdversarialQA validation

| 학습 데이터 / 형식 / loss | Single | Multi gold history | Multi rollout |
|---|---:|---:|---:|
| Base (no fine-tuning) | 0.25 | 0.22 | 0.19 |
| SQuAD / single / assistant-only | 0.23 | 0.21 | 0.21 |
| SQuAD / single / system loss | 0.18 | 0.20 | 0.18 |
| SQuAD / single / user loss | 0.22 | 0.21 | 0.16 |
| SQuAD / single / all loss | 0.23 | 0.21 | 0.16 |
| SQuAD / multi / assistant-only | 0.24 | 0.24 | 0.22 |
| SQuAD / multi / system loss | 0.22 | 0.25 | 0.22 |
| SQuAD / multi / user loss | 0.20 | 0.20 | 0.20 |
| SQuAD / multi / all loss | 0.21 | 0.23 | 0.20 |
| AdversarialQA / single / assistant-only | **0.29** | **0.25** | **0.24** |
| AdversarialQA / single / system loss | 0.22 | 0.23 | 0.20 |
| AdversarialQA / single / user loss | 0.27 | 0.19 | 0.18 |

**초기 해석:** SQuAD 평가에서는 SQuAD로 학습한 조건의 점수가 대체로 base보다 높았고, 최고 single EM은 multi/assistant-only의 0.74였습니다. 하지만 multi/system-loss가 multi-turn 지표에서 가장 높았으므로 assistant-only가 일관된 승자라고 할 수는 없습니다. AdversarialQA 평가에서는 SQuAD 학습 조건의 이득이 일관되지 않았으며, 데이터셋 내 비교도 완료된 세 조건뿐입니다. 따라서 데이터 혼합 효과나 24개 조건 전체의 우열은 이 부분 실행만으로 판단할 수 없습니다.

Single 대 multi 형식의 SQuAD-trained assistant-only 직접 비교에서 EM 차이(multi − single)는 SQuAD 평가에서 +0.02 (single), +0.05 (multi gold), +0.04 (rollout)이었습니다. passage-cluster bootstrap 95% CI는 각각 [-0.04, 0.08], [-0.01, 0.12], [-0.02, 0.10]으로 모두 0을 포함합니다. 형식의 우위를 확인했다고 볼 수 없습니다. 한 seed·100문항 평가이므로 모든 관찰은 탐색적이며, prompt loss 비교는 prompt 토큰을 추가하는 동시에 assistant 토큰의 상대적 loss 비중도 바꾸는 비교입니다.

전체 11개 조건의 EM, paired bootstrap 비교 및 형식/objective별 결과는 [요약 JSON](results/prompt-mask-ablation-summary.json), [mask 비교](results/prompt-mask-ablation-paired-comparisons.json), [single/multi 비교](results/prompt-mask-ablation-turn-format-comparison.json), [assistant-only/all-loss 비교](results/prompt-mask-ablation-objective-comparison.json)에 있습니다. 이 결과는 위의 Qwen2.5-1.5B SQuAD 파일럿 보고서와 별개의 실험입니다.

### 후속 형식: 매 턴 지문 재제공

원래 multi 형식은 첫 user 턴에만 지문을 제공하고 이후 독립 질문을 이어 붙입니다. 앞선 QA 쌍이 뒤 질문에 대한 in-context 예시처럼 기능할 수 있다는 점을 확인하기 위해 **각 user 턴에 지문과 해당 질문을 함께 제공**하는 형식을 추가하고 1-epoch 파일럿을 실행했습니다. 공통 데이터는 100 train groups와 20 held-out groups (각각 400/80 QA targets), Qwen2.5-0.5B-Instruct, seed 42, assistant-only loss입니다. 두 형식 모두 optimizer update 100회이며, 기본 형식 대비 지문 반복 형식은 입력 토큰이 31,390에서 83,683으로 증가했습니다.

| 학습 형식 | Multi gold EM / F1 | Multi rollout EM / F1 | Single EM / F1 |
|---|---:|---:|---:|
| Base | 0.588 / 0.743 | 0.575 / 0.726 | 0.613 / 0.733 |
| Original multi (passage only on turn 1) | **0.738 / 0.846** | 0.688 / 0.808 | 0.713 / 0.825 |
| Repeat-passage multi (passage on every turn) | 0.713 / 0.827 | **0.700 / 0.843** | 0.713 / 0.827 |

두 학습 형식의 F1 차이(repeat-passage minus original)는 gold multi -0.019, rollout multi +0.018, single +0.002였고, 각 passage-cluster bootstrap 95% CI가 모두 0을 포함했습니다(20개 passage). 따라서 이 작은 파일럿에서는 지문 반복 형식이 더 낫거나 나쁘다고 결론낼 수 없습니다. 반복 제공이 해당 질문에서 정답 근거를 다시 볼 수 있게 하는 대신, 입력 토큰/FLOPs를 크게 늘린다는 점도 비교 해석에 유의해야 합니다. SQuAD 질문은 독립적이므로 실제 history-dependent 대화 평가가 아닙니다.

학습/평가 파일은 [파일럿 출력 디렉터리](results/prompt_mask_repeat_passage/)에 있습니다. 이 결과는 앞의 1,000-target 0.5B 부분 ablation 및 기존 1.5B 파일럿과 별도이며, 작은 샘플·단일 seed의 탐색 결과입니다.
