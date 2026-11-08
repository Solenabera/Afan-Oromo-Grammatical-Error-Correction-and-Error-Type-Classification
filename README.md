# AO-GEC: Afaan Oromoo Grammatical Error Correction & Classification

This repository contains experimental research on **Grammatical Error Correction (GEC)** and **Error Type Classification** for **Afaan Oromoo**, a Cushitic language spoken by over 35 million people across East Africa.

While modern GEC systems work well for high-resource languages like English or German, morphologically rich and low-resource languages like Afaan Oromoo present unique challenges: complex suffixation patterns, SOV word order nuances, and a scarcity of standardized parallel error corpora. This project investigates whether joint multi-task learning—combining error diagnosis with sequence-to-sequence correction—improves both correction quality and model interpretability.

---

## Key Ideas & Approach

Standard end-to-end GEC models treat error correction as a black-box sequence translation problem (`faulty sentence -> corrected sentence`). In low-resource settings, this often leads to hallucinated paraphrases or failure to address the underlying grammatical issue.

Here, we explore two complementary angles:
1. **Error Diagnosis (Classification):** Can a model identify *what* kind of mistake was made (e.g., subject-verb disagreement vs. word order)?
2. **Conditioned Correction (Generation):** Does explicitly knowing (or predicting) the error type help the decoder make more precise, surgical corrections?

To explore this, we implemented three distinct pipelines:
- **Baseline Generation:** Pure Seq2Seq T5 correction without auxiliary signals.
- **Baseline Classifier:** T5 encoder + pooling head dedicated to 4-way error taxonomy classification.
- **Joint Multi-Task T5 (`T5MultiTask`):** A dual-stream architecture with shared encoder representations trained in a staged curriculum to diagnose and fix errors simultaneously.

---

## Architecture Overview

<p align="center">
  <img src="Proposed%20Model%20Architecture.eps" alt="Proposed Model Architecture" width="90%"/>
  <br>
  <em>Figure 1: High-level schematic of the proposed dual-stream T5 multi-task model (vector source: <a href="Proposed%20Model%20Architecture.eps"><code>Proposed Model Architecture.eps</code></a>).</em>
</p>

In the joint multi-task model (`ao-gec-generation-and-classifier-using-t5.ipynb`), a single shared encoder powers two task streams:

```
                          [ Input Sentence ]
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
   Neutral Prompt                                   Labeled Prompt
"classify error: {text}"                       "correct {error} error: {text}"
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  ▼
                         Shared T5 Encoder
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
   Encoder States                                   Encoder States
         │                                                 │
  Mean + Max Pool                                          │
         │                                                 │
   Deep MLP Head                                      T5 Decoder
  (GELU + LayerNorm)                                       │
         ▼                                                 ▼
 Predicted Error Type                           Corrected Afaan Oromoo
  (Class Focal Loss)                           (Smoothed Cross-Entropy)
```

### Dual-Stream Mechanics
- **Stream A (Diagnostic / Neutral Stream):** Takes a neutral prompt without leaking the ground-truth error type. Encoder hidden states are pooled using concatenated mean and max pooling (`2 * d_model`), passing through an MLP with GELU activations and LayerNorm to predict the error category.
- **Stream B (Correction / Labeled Stream):** Takes an error-conditioned prompt (`"correct the {error_type} error in afan oromo: {text}"`). During training, it uses the true label; at test time, it can be conditioned on the label predicted by Stream A.

### Staged Training Curriculum
To prevent the decoder from destabilizing the encoder representations early on:
1. **Stage 0 (Classifier Warm-up):** The T5 decoder is frozen. The model optimizes only the classification head and encoder on the diagnostic task.
2. **Stage 1 (Joint Multi-Task Training):** Both streams train simultaneously with a joint objective:
   $$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{gen} + \beta \cdot \mathcal{L}_{focal\_cls}$$
   Class-balanced focal loss ($\gamma = 1.6$) prevents majority error types from dominating.
3. **Stage 2 (Refinement):** Generation-focused fine-tuning with early stopping monitored on validation GLEU.

---

## Error Taxonomy

The dataset categorizes errors into four common grammatical pitfalls in Afaan Oromoo writing:

| Error Type | Description |
| :--- | :--- |
| `subject-verb-agreement-error` | Mismatched person, number, or gender inflections between the subject and the verb. |
| `noun-adjective-agreement-error` | Gender/number agreement inconsistencies between head nouns and their modifying adjectives. |
| `adverb-verb-agreement-error` | Incorrect adverbial modifiers or tense/aspect incompatibilities with the predicate. |
| `word-order-error` | Deviations from natural Afaan Oromoo constituent ordering (SOV / post-positional modifiers). |

---

## Repository Structure

```text
AO-GEC-Research/
├── Proposed Model Architecture.eps                 # Publication-ready vector diagram (Figure 1)
├── ao-gec-using-t5-generation-baseline.ipynb       # Baseline Seq2Seq GEC (T5)
├── ao-gec-using-t5-classifier-baseline.ipynb       # Baseline 4-way error classifier
├── ao-gec-generation-and-classifier-using-t5.ipynb # Dual-stream joint multi-task model
└── README.md
```

### Notebook Details

1. **`ao-gec-using-t5-classifier-baseline.ipynb`**
   - Implements text normalization, pooling over T5 encoder tokens, and class-weighted cross-entropy.
   - Evaluates on Macro F1, Precision, Recall, and Confusion Matrices.
   - *Test Results:* Reaches **~90.0% accuracy** and **0.90 Macro F1** across the 4 error categories.

2. **`ao-gec-using-t5-generation-baseline.ipynb`**
   - Straightforward T5 sequence-to-sequence baseline fine-tuned to map erroneous sentences to clean sentences.
   - Evaluated with corpus-level and sentence-level **GLEU** (with copy-penalty following Napoles et al., 2015) and **chrF**.

3. **`ao-gec-generation-and-classifier-using-t5.ipynb`**
   - The core dual-stream model combining both tasks.
   - Includes pre-tokenized dual-view data loading, mixed precision (bf16/fp16), gradient checkpointing, and staged warm-up routines.

---

## Evaluation Metrics

Evaluating GEC requires metrics sensitive to minor token edits while penalizing spurious alterations:
- **GLEU (Generalized Language Evaluation Understanding):** The standard metric for GEC (Napoles et al., 2015). It computes n-gram precision over both the candidate and reference sentences with an explicit penalty for leaving erroneous n-grams untouched or hallucinating edits.
- **chrF:** Character n-gram F-score, particularly valuable for Afaan Oromoo due to extensive agglutinative morphology and suffix changes.
- **Classification Macro F1 & Confusion Matrix:** Ensures balanced diagnostic accuracy across all error classes.

---

## Experimental Setup & Quickstart

The experiments are designed to run in notebook environments (such as Kaggle with a T4 or A100 GPU, or a local CUDA machine).

### 1. Requirements

```bash
pip install torch transformers datasets accelerate evaluate scikit-learn pandas numpy matplotlib seaborn wandb tqdm
```

### 2. Dataset Setup
The dataset used in these notebooks is hosted on Kaggle:
- Dataset: `solanadh/aogec-dataset-with-error-type-v03`
- Expected splits:
  - `AOGEC_Dataset_Labeled_train.csv`
  - `AOGEC_Dataset_Labeled_eval.csv`
  - `AOGEC_Dataset_Labeled_test.csv`
- Each split contains three primary columns: `input`, `target`, and `error_type`.

If running locally, update the `data_dir` attribute inside the `CFG` class in each notebook:
```python
@dataclass
class CFG:
    data_dir: str = "./data"  # path to directory containing CSV splits
    model_name: str = "t5-base"
    batch_size: int = 8
    grad_accum_steps: int = 4
    amp_dtype: str = "bf16"   # "bf16" for Ampere/Ada/Hopper, "fp16" for T4/V100
```

### 3. Running Experiments
You can open and execute the notebooks sequentially:
1. Run `ao-gec-using-t5-classifier-baseline.ipynb` to inspect classification difficulty and error confusion.
2. Run `ao-gec-using-t5-generation-baseline.ipynb` to establish the seq2seq benchmark.
3. Run `ao-gec-generation-and-classifier-using-t5.ipynb` for the full dual-stream multi-task model.

---

## Observations & Findings

- **Error Classification is Feasible:** The baseline classifier achieves strong separation (~95% F1 on agreement errors, ~87% F1 on word order), indicating that the T5 encoder captures syntactic cues in Afaan Oromoo even without dedicated Cushitic language pretraining.
- **Subject-Verb Agreement is the Hardest Class:** Subject-verb agreement shows slightly lower recall (~81%) than noun-adjective or adverb-verb pairs, largely due to pro-drop characteristics and long-distance dependencies in complex sentences.
- **Conditioning Stabilizes Corrections:** Passing the error category into the generator helps prevent the model from over-editing or rewriting valid phrasing, keeping changes localized to the true error locus.

---

## Roadmap & Next Steps

- [ ] **Multilingual Encoders:** Evaluate `google/mt5-base` and `google/byt5-base` to test whether byte-level or multilingual Cushitic token representations outperform standard `t5-base`.
- [ ] **End-to-End Inference Chaining:** Benchmark test-set generation quality when Stream B is conditioned on Stream A's *predicted* error type vs. ground truth (pipelined inference).
- [ ] **Multi-Error Sentences:** Extend the taxonomy and dataset to handle sentences containing more than one simultaneous error.
- [ ] **Synthetic Data Augmentation:** Develop rule-based error injection scripts tailored to Afaan Oromoo morphology to augment the training data volume.

---

## Citation & Acknowledgments

If you find this research or dataset useful in your work, feel free to cite or reference this repository:

```bibtex
@misc{aogec2026,
  author = {Solana Dh.},
  title = {AO-GEC: Afaan Oromoo Grammatical Error Correction and Error Classification with T5},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/solanadh/AO-GEC-Research}}
}
```

*Special thanks to the open-source NLP and Ethiopian language computing communities working to make low-resource African languages first-class citizens in modern AI.*

