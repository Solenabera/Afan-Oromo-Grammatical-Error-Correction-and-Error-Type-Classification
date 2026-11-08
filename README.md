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
- **Joint Multi-Task T5 (`T5MultiTask`):** A dual-stream architecture with shared encoder representations trained in a staged curriculum to diagnose and fix errors simultaneously, evaluated with GLEU and Max-Match ($M^2$).

---

## Architecture Overview

<p align="center">
  <img src="Proposed%20Model%20Architecture.png" alt="Proposed Model Architecture" width="90%"/>
  <br>
  <em>Figure 1: High-level schematic of the proposed dual-stream T5 multi-task model (<code>Proposed Model Architecture.png</code>).</em>
</p>

In the joint multi-task model (`ao-gec-generation-and-classifier-m2-t5multitask.ipynb`), a single shared encoder powers two task streams:

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
- **Stream A (Diagnostic / Neutral Stream):** Takes a neutral prompt (`"classify the grammatical error in afan oromo: {text}"`) without leaking the ground-truth error type. Encoder hidden states are pooled using concatenated mean and max pooling (`2 * d_model`), passing through an MLP with GELU activations and LayerNorm to predict the error category.
- **Stream B (Correction / Labeled Stream):** Takes an error-conditioned prompt (`"correct the {error_type} error in afan oromo: {text}"`). During training, it uses the true label; at test time, it can be dynamically conditioned on the label predicted by Stream A.

### Staged Training Curriculum
To prevent the decoder from destabilizing the encoder representations early on:
1. **Stage 0 (Classifier Warm-up — 4 Epochs):** The T5 decoder is frozen. The model optimizes only the classification head and encoder on the diagnostic task ($\beta_{cls} = 1.2$).
2. **Stage 1 (Joint Multi-Task Training — 10 Epochs):** Both streams train simultaneously with a joint objective:
   $$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{gen} + \beta \cdot \mathcal{L}_{focal\_cls}$$
   Class-balanced focal loss ($\gamma = 1.6$) prevents majority error types from dominating. The best checkpoint is selected via the harmonic mean of validation Macro F1 and normalized GLEU.
3. **Stage 2 (Generation Refinement — 3 Epochs):** The classification loss weight is reduced ($\beta_{cls} = 0.12$) to prioritize generation fluency and surgical corrections.

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
├── Proposed Model Architecture.png                 # Architecture schematic (Figure 1)
├── ao-gec-using-t5-classifier-baseline.ipynb       # Baseline 4-way error classifier
├── ao-gec-using-t5-generation-baseline.ipynb       # Baseline Seq2Seq GEC (T5)
├── ao-gec-generation-and-classifier-m2-t5multitask.ipynb # Dual-stream joint multi-task model (GLEU + M²)
└── README.md
```

### Notebook Details

1. **`ao-gec-using-t5-classifier-baseline.ipynb`**
   - Standalone diagnostic baseline using T5 encoder representations and a classification head.
   - Evaluates on Macro F1, Precision, Recall, and Confusion Matrices.
   - *Test Results:* Reaches **~90.0% accuracy** and **0.90 Macro F1** across the 4 error categories.

2. **`ao-gec-using-t5-generation-baseline.ipynb`**
   - Standard T5 sequence-to-sequence baseline fine-tuned to map erroneous sentences to clean sentences without error conditioning.
   - Evaluated with corpus-level and sentence-level **GLEU** and **chrF**.
   - *Test Results:* GLEU: **13.47**, chrF: **86.03**.

3. **`ao-gec-generation-and-classifier-m2-t5multitask.ipynb`**
   - The primary dual-stream multi-task model (`T5MultiTask`) combining diagnosis and correction.
   - Implements both **GLEU** and **$M^2$ (Max-Match)** scorers alongside **chrF**.
   - Implements pre-tokenized dual-view data loading, mixed precision (`bf16`/`fp16`), gradient checkpointing, and the 3-stage curriculum.
   - *Validation Progress:*
     - Stage 0 (Warm-up): Macro F1 reaches **0.6901** (Acc: 73.06%).
     - Stage 1 (Joint Training): Macro F1 reaches **0.9076**, GLEU reaches **67.98**, chrF reaches **91.66**.
     - Stage 2 (Refinement): **GLEU reaches 70.41**, **chrF reaches 92.07**, Macro F1 stabilizes at **0.9102**.

---

## Evaluation Metrics

Evaluating GEC requires metrics sensitive to minor token edits while penalizing spurious alterations:
- **GLEU (Generalized Language Evaluation Understanding):** The standard metric for GEC (Napoles et al., 2015). It computes n-gram precision over both the candidate and reference sentences with an explicit penalty for leaving erroneous n-grams untouched or hallucinating edits.
- **$M^2$ (Max-Match Scorer — $F_{0.5}$):** Computes edit-level precision ($P$), recall ($R$), and $F_{0.5}$ (Dahlmeier & Ng, 2012) via sequence diff opcodes. By setting $\beta = 0.5$, it weights precision twice as heavily as recall to reward models that make accurate, conservative edits rather than over-correcting.
- **chrF:** Character n-gram F-score, particularly valuable for Afaan Oromoo due to extensive agglutinative morphology and suffix changes.
- **Classification Macro F1 & Confusion Matrix:** Ensures balanced diagnostic accuracy across all error classes.

---

## End-to-End Pipelined Inference

The joint model supports realistic two-pass inference via `predict_single()`:
1. **Step 1:** Feeds the uncorrected sentence to Stream A using a neutral prompt to classify the error type.
2. **Step 2:** Formulates an error-conditioned prompt using the predicted label and generates the correction with Stream B using beam search.

```
Raw Input: "Ijaarani jiru isaan mana haaraa."
    │
    ▼ (Stream A: Neutral Prompt)
Predicted Error: word-order-error
    │
    ▼ (Stream B: "correct the word-order-error error in afan oromo: ...")
Corrected Output: "isaan mana haaraa Ijaarani jiru."
```

### Demo Samples

| Input Erroneous Sentence | Predicted Error | Model Correction |
| :--- | :--- | :--- |
| `Abubakir fi Galgaloon bara kana barataa cimoo jedhamuu badhaafame turre .` | `subject-verb-agreement-error` | `Abubakir fi Galgaloon bara kana barattoota cimoo jedhamuun badhaafamanii turan.` |
| `Ijaarani jiru isaan mana haaraa.` | `word-order-error` | `isaan mana haaraa Ijaarani jiru.` |
| `Isheen kolfanii hojii ishee xumurte.` | `adverb-verb-agreement-error` | `isheen kolfitee hojii ishee xumurte.` |
| `Inistiitiyuutiin Siinqee dubartootaa guyyaa idil addunyaa kabaje .` | `subject-verb-agreement-error` | `Inistiitiyuutiin Siinqee dubartootaa guyyaa idil addunyaan kabaje.` |

---

## Experimental Setup & Quickstart

The experiments are designed to run in notebook environments (such as Kaggle with a T4 or A100 GPU, or a local CUDA machine).

### 1. Requirements

```bash
pip install torch transformers datasets accelerate evaluate scikit-learn pandas numpy matplotlib seaborn wandb tqdm
```

### 2. Dataset Setup
The dataset used in these notebooks is hosted on Kaggle:
- Dataset: `solanadh/ao-gec-dataset-with-error-type` (or `solanadh/aogec-dataset-with-error-type-v03`)
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
3. Run `ao-gec-generation-and-classifier-m2-t5multitask.ipynb` for the full dual-stream multi-task model with GLEU and $M^2$ evaluation.

---

## Observations & Findings

- **Multi-Task Synergies Outperform Pure Generation:** The pure seq2seq baseline achieves a GLEU of ~13.5, while the joint dual-stream model achieves a validation GLEU of **70.41** and chrF of **92.07**. Conditioning the generation decoder on explicit grammatical error categories dramatically guides the search space.
- **Error Classification is Highly Separable:** The classifier head reliably achieves ~91% accuracy and 0.91 Macro F1 across four core grammatical categories.
- **Agreement vs. Word Order:** Agreement errors benefit heavily from token-level morphological alignment, whereas word-order errors require the decoder to reorganize syntactic constituents while preserving roots.

---

## Roadmap & Next Steps

- [ ] **Multilingual Pretrained Encoders:** Benchmark `google/mt5-base` and `google/byt5-base` to evaluate whether subword/byte-level representations better capture Cushitic agglutination.
- [ ] **Full Multi-Error Sentences:** Expand dataset annotations and loss formulations to handle complex sentences containing multiple concurrent error types.
- [ ] **Morphology-Guided Data Augmentation:** Build linguistic perturbation rules for Afaan Oromoo to expand training coverage for rare grammatical forms.

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
