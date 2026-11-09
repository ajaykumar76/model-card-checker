# Model Card Documentation Checker (tested on real Hugging Face model cards)

Checks whether a model card documents 8 elements that regulators and model-card practice ask for:
**intended use, out-of-scope use, limitations, bias and risks, training data, evaluation, training compute /
environment, contact.** The elements are inspired by the model-card literature and by the documentation
themes in EU AI Act Annex IV. They are **not** a legal checklist and this tool is **not** compliance advice.

> **Short version of the result:** on 40 hand-labelled real cards, a local 3B LLM (with a rule that its
> evidence must be an exact quote from the card) and a plain keyword baseline reach **similar accuracy
> (81% vs 78% on test)**. They fail in opposite ways. Keywords find most real documentation but call a
> missing section "documented" **24%** of the time; the LLM does that **9%** of the time but **misses
> about 45%** of the documentation that exists. Neither is safe to use unsupervised.

## The question

When a card says nothing about, say, evaluation, the dangerous mistake is **false assurance**: the checker
says "documented". The evaluation reports that separately from accuracy.

## How it works

1. **Rules** (`--method rules`): keyword and regex baseline per element, ignoring placeholders ("More Information Needed", TODO).
2. **LLM** (`--method llm`): split the card into paragraphs (code blocks removed), embed them
   (`all-MiniLM-L6-v2`), take the 6 best-matching paragraphs per element, and ask a local `llama3.2`
   (via Ollama) for a yes/no plus a quote.
3. **Guardrail** (`--method llmchk`): a "documented" answer only stands if its quote is an exact,
   non-placeholder span of the card. This overrode 20 of the model's answers on dev and 15 on test.

## Data and labels (read this before the numbers)

- 172 public model cards collected from the Hugging Face Hub (seeded sample over popularity tiers;
  front matter removed). **40 were hand-labelled** (20 dev, 20 test) on 8 binary elements = 320 labels in total.
- **The labels were written by Claude (an AI assistant) from the rubric in `data/rubric.md`.** I (the project
  owner) have **not** independently reviewed them, so they are one reviewer's judgement, not ground truth.
  75 of 320 labels are flagged *borderline*; results are shown with and without them.
- Rules for a "1": placeholders, empty headings and bare links do not count; images cannot be read.
- Cards longer than 12,000 characters (62 of 172) were **not** in the labelled sample, so the checker is unvalidated on them.
- Card texts are not in this repo (licences differ). Run `src/01_collect_cards.py` to re-collect; because
  the hub's download ranking changes, a re-run will not give the identical sample. Labels are keyed by model id.

## Results

95% Wilson intervals. Items on the same card are correlated, so intervals are **optimistically narrow**.

### Dev (20 cards, 160 items), two prompt rounds

| | Rules | LLM v1 + check | LLM v2 + check |
|---|---|---|---|
| Accuracy | 73% | 74% | 76% |
| Real documentation found | 76% | 18% | 53% |
| False assurance | 28% | 0% | 14% |

- **v1** told the model to be "strict". It said "documented" only 9 times in 160 (recall 18%), close to a
  checker that always answers "no" (68% accuracy). Emphasis made a small model over-correct, as in my earlier projects.
- **v2** used a neutral prompt, the "does not count" cases inside the element definition, and 6 excerpts instead of 4.
- **Rule fixed before the test run:** freeze the LLM only if it beats rules on dev accuracy *and* false assurance. v2 did, narrowly.

### Test (20 cards, 160 items), run once

| | Rules | LLM v2 + check |
|---|---|---|
| Accuracy | 124/160 = 78% (70-83%) | 130/160 = 81% (74-87%) |
| Precision of "documented" | 35/63 = 56% | 24/35 = 69% |
| Real documentation found | 35/43 = 81% (67-90%) | 24/43 = 56% (41-70%) |
| **False assurance** | 28/117 = **24%** (17-32%) | 11/117 = **9%** (5-16%) |
| Accuracy, clear labels only | 83% | 86% |

Always answering "not documented" would score 73% on test, so both methods beat the trivial baseline by only
5-8 points. Rules are **weakest on evaluation, compute and contact** (they match "results", "hours", "support" in unrelated text:
7/14, 8/17 and 5/17 gaps wrongly called documented). The LLM is **weakest on intended use and limitations**
(it treats a general description of the model as an intended-use statement).

## What the data says about real cards (rough, not a prevalence estimate)

Run on all 172 cards. The two methods disagree a lot, so I report ranges, not one number:

| Element | Rules | LLM + check |
|---|---|---|
| Out-of-scope use | 15% | 16% |
| Limitations | 35% | 38% |
| Bias / risks | 33% | 17% |
| Evaluation | 78% | 45% |
| Training data | 65% | 35% |
| Training compute / environment | 68% | 23% |
| Contact | 64% | 15% |
| Intended use | 47% | 71% (the LLM over-calls this one) |

Rules over-count and the LLM under-counts, so the truth is plausibly between them for most elements.
Only the first two rows, where the methods agree, support a firm statement: **explicit out-of-scope-use
sections appear in only about 1 in 6 cards.** Everything else is a range.

## What I take from this

- **Neither method is reliable enough to certify a card.** The useful pattern is two-sided: use rules as a cheap
  *candidate finder* (high recall) and a human or stronger check to confirm. I did not test that combination.
- **The exact-quote check is the most valuable part.** It removed 15-20 unsupported "documented" claims per split at no cost in recall.
- **Accuracy hides the story.** Both methods look similar on accuracy and differ 2-3x on false assurance.
- Differences in accuracy between the two methods (3 points) are within noise on 20 cards.

## Limitations

- Labels by one AI reviewer, not independently verified; spot-check pending.
- 40 labelled cards, correlated items, one small model, one embedding model.
- Text-only: charts and badges are invisible; cards over 12,000 characters are unvalidated.
- Dev was used for two prompt rounds; the version was fixed before the single test run.
- The rule set was written once and not tuned on dev.
- The population table is not a measured prevalence.

## Run it

```powershell
pip install -r requirements.txt
ollama pull llama3.2
python src\01_collect_cards.py                                              # collects cards (needs internet)
python src\02_check_eval.py --split dev --method rules
python src\02_check_eval.py --split dev --method llmchk --tag v2 --prompt v2
python src\02_check_eval.py --split test --method llmchk --tag v2 --prompt v2   # run once, at the end
python src\02_check_eval.py --population --method llmchk --tag v2 --prompt v2
```

## Files

```
src/01_collect_cards.py   collects public cards, picks a sample to label
src/02_check_eval.py      rules, LLM, guardrail, evaluation, population run
data/rubric.md            labelling rubric (8 elements)
data/gold_labels.json     the 40 hand labels (keyed by model id)
reports/                  saved results of each run
```

