# DST5-GECC: Dual-Stream T5 for Afaan Oromoo Grammatical Error Correction & Classification

This repository contains experimental research on **Grammatical Error Correction (GEC)** and **Error Type Classification** for **Afaan Oromoo**, a Cushitic language spoken by over 60 million people across East Africa.

While modern GEC systems work well for high-resource languages like English or German, morphologically rich and low-resource languages like Afaan Oromoo present unique challenges: complex agglutinative suffixation, subject-object-verb (SOV) syntactic structures, and a scarcity of standardized parallel error corpora. This project introduces **`DST5-GECC`** (**D**ual-**S**tream **T5** for **G**rammatical **E**rror **C**orrection and **C**lassification), a unified multi-task framework that jointly diagnoses error types and conditions the generative decoder to produce precise, surgical corrections.

---

## Key Ideas & Approach

Standard end-to-end GEC models treat error correction as a black-box sequence-to-sequence translation problem (`erroneous sentence -> corrected sentence`). In low-resource settings, this often leads to hallucinated paraphrases or failure to localize the underlying grammatical defect.

**`DST5-GECC`** explores two complementary angles:
1. **Error Diagnosis (Stream A — Classification):** Inferring *what* type of mistake was made (e.g., subject-verb agreement vs. word order) from unannotated text.
2. **Conditioned Correction (Stream B — Generation):** Conditioning the decoder on the diagnosed error type to guide the model toward targeted, conservative edits rather than wholesale rewriting.

To benchmark this approach, we evaluate three experimental setups:
- **Baseline 1 (Classification Only):** Dedicated single-task classifier utilizing T5 encoder representations with a pooled multi-layer MLP head.
- **Baseline 2 (Generation Only):** Standard sequence-to-sequence T5 model trained to map faulty inputs directly to clean outputs without auxiliary signals.
- **Proposed `DST5-GECC` (Ours):** Dual-stream multi-task model sharing encoder representations across both tasks, trained via a staged curriculum and evaluated with GLEU, chrF, and multi-class classification metrics.

---

## Architecture Overview

<p align="center">
  <img src="Proposed%20Model%20Architecture.png" alt="Proposed Model Architecture" width="90%"/>
  <br>
  <em>Figure 1: High-level schematic of the proposed dual-stream DST5-GECC multi-task model (<code>Proposed Model Architecture.png</code>).</em>
</p>

In the `DST5-GECC` architecture (`ao-gec-generation-and-classifier-using-t5.ipynb`), a single shared encoder powers both task streams:

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
- **Stream A (Diagnostic / Neutral Stream):** Ingests an unannotated prompt (`"classify the grammatical error in afan oromo: {text}"`). Token representations are pooled using concatenated mean and max pooling ($2 \times d_{model}$), then passed through a 4-layer MLP with GELU activations and LayerNorm to predict one of four error categories via class-balanced Focal Loss ($\gamma = 1.6$).
- **Stream B (Correction / Labeled Stream):** Ingests an error-conditioned prompt (`"correct the {error_type} error in afan oromo: {text}"`). During training, it is conditioned on the gold label; during inference, it dynamically conditions on Stream A's predicted category to decode corrected text using beam search.

### Staged Training Curriculum
To prevent early decoder gradients from degrading the shared encoder's representation space:
1. **Stage 0 (Classifier Warm-up — 4 Epochs):** The T5 decoder is frozen. The model optimizes only the classification head and encoder on the diagnostic task ($\beta_{cls} = 1.2$).
2. **Stage 1 (Joint Multi-Task Training — 10 Epochs):** Both streams train simultaneously with a joint objective:
   $$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{gen} + \beta \cdot \mathcal{L}_{focal\_cls}$$
   The best checkpoint is tracked by the harmonic mean of validation Macro $F_1$ and normalized GLEU.
3. **Stage 2 (Generation Refinement — 3 Epochs):** The classification weight is scaled down ($\beta_{cls} = 0.12$) to refine beam search decoding fluency and output calibration.

---

## Error Taxonomy

The dataset targets four primary grammatical error categories encountered in Afaan Oromoo writing:

| Error Type | Description | Linguistic Challenge in Afaan Oromoo |
| :--- | :--- | :--- |
| `subject-verb-agreement-error` | Disagreements in person, number, or gender inflections between the grammatical subject and predicate. | Pro-drop constructions and long-distance constituent dependencies. |
| `noun-adjective-agreement-error` | Inconsistent gender/number suffixes between head nouns and modifying adjectives. | Suffix harmony and gender marking rules across noun phrases. |
| `adverb-verb-agreement-error` | Erroneous adverbial modifiers or tense/aspect incompatibilities with the predicate verb. | Verb conjugation tense/aspect particle selection. |
| `word-order-error` | Constituent displacement violating canonical SOV ordering or post-positional modifier rules. | Re-ordering multi-word phrases without corrupting stem morphology. |

---

## Evaluation Metrics

We evaluate model performance across both diagnostic (classification) and generative (correction) tasks using two primary families of evaluation metrics: **Multi-Class Classification Metrics** and **General Language Evaluation Understanding (GLEU)**.

### 1. Multi-Class Classification Metrics (Stream A)

To evaluate the classifier's diagnostic capabilities, we employ Macro-averaged Precision ($P_{\text{cls}}$), Macro-averaged Recall ($R_{\text{cls}}$), and the standard Macro-$F_1$ score ($F_1$). The primary advantage of macro-averaging over micro-averaging is that it treats every grammatical error class equally, ensuring that dominant or highly frequent categories do not artificially inflate overall performance scores:

$$
P_{\text{cls}} = \frac{1}{|C|} \sum_{c \in C} \frac{TP_c}{TP_c + FP_c}
$$

$$
R_{\text{cls}} = \frac{1}{|C|} \sum_{c \in C} \frac{TP_c}{TP_c + FN_c}
$$

$$
F_1 = \frac{1}{|C|} \sum_{c \in C} \frac{2 \times P_c \times R_c}{P_c + R_c}
$$

where true positives ($TP_c$), false positives ($FP_c$), false negatives ($FN_c$), precision ($P_c$), and recall ($R_c$) are computed locally for each error class $c \in C$.

### 2. General Language Evaluation Understanding (GLEU — Stream B)

We use the **General Language Evaluation Understanding (GLEU)** metric to evaluate model capability in the generative correction task. GLEU assesses the $n$-gram overlap between the model's hypothesis ($H$) and reference corrections ($R$) relative to the source input ($S$).

Unlike standard machine translation metrics such as BLEU, GLEU is specifically designed for grammatical error correction: it rewards correct syntactic edits while explicitly penalizing the model for leaving errors unchanged or introducing false edits. Given source sentences $S$, candidate hypotheses $H$, and reference targets $R$, GLEU is defined as:

$$
\text{GLEU}(S, H, R) = \text{BP} \times \exp\left( \frac{1}{N} \sum_{k=1}^{N} \log P_k \right)
$$

where $N = 4$ represents the maximum $n$-gram order.

#### Brevity Penalty ($\text{BP}$)
To prevent the model from generating artificially short or truncated sentences, the Brevity Penalty is applied:

$$
\text{BP} = 
\begin{cases} 
  1 & \text{if } l_h > l_r, \\
  \exp\left(1 - \frac{l_r}{l_h}\right) & \text{if } l_h \leq l_r
\end{cases}
$$

where $l_h$ and $l_r$ represent the total token counts for the hypothesis and reference sequences, respectively.

#### Modified $n$-gram Precision ($P_k$)
The modified $n$-gram precision adjusts standard precision by isolating genuine grammatical improvements over the original source text:

$$
P_k = \frac{\sum_{ng \in H} \left[ C_R(ng) - \max\left(0, C_S(ng) - C_R(ng)\right) \right]}{\sum_{ng \in H} C_H(ng)}
$$

where $C_X(ng)$ represents the count of a specific $n$-gram $ng$ within sequence $X \in \{S, H, R\}$.

> [!NOTE]
> In addition to GLEU, we report character $n$-gram F-score (**chrF**) in our benchmark results to capture subword and morpheme-level precision, which is particularly informative given Afaan Oromoo's rich agglutinative inflection system.

---

## Results and Discussion

We evaluated the proposed **`DST5-GECC`** architecture against both single-task baselines on the held-out test dataset. Stream A's error classification accuracy was compared against the dedicated single-task Baseline 1, and Stream B's generative correction performance was evaluated against the standard sequence-to-sequence Baseline 2.

### Comprehensive Performance Comparison on the Test Dataset

| Model Architecture | Classification (Stream A)<br>Precision ($P_{\text{cls}}$) | Classification (Stream A)<br>Recall ($R_{\text{cls}}$) | Classification (Stream A)<br>Macro-$F_1$ ($F_1$) | Generation (Stream B)<br>chrF | Generation (Stream B)<br>GLEU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1** *(Classification Only)* | **90.15%** | **90.31%** | **90.23%** | — | — |
| **Baseline 2** *(Generation Only)* | — | — | — | 91.29 | 66.27 |
| **Proposed `DST5-GECC`** *(Ours)* | 89.32% | 89.93% | 89.62% | **91.98** | **68.81** |

### Key Findings & Analysis

1. **Superior Generative Correction (+2.54 GLEU, +0.69 chrF):**
   - The proposed **`DST5-GECC`** achieves a test **GLEU of 68.81** and **chrF of 91.98**, significantly outperforming the standard sequence-to-sequence baseline (**66.27 GLEU** and **91.29 chrF**).
   - Providing explicit error-type conditioning in Stream B constrains the decoder hypothesis space, discouraging hallucinated paraphrasing and encouraging precise grammatical repairs.

2. **Negligible Multi-Task Penalty on Classification (89.62% vs. 90.23% Macro-$F_1$):**
   - While joint multi-task models often suffer from negative task interference ("multi-task tax"), Stream A of `DST5-GECC` retains **89.62% Macro-$F_1$** (89.32% precision, 89.93% recall)—less than 0.61% behind the dedicated, unconstrained Baseline 1.
   - Sharing encoder weights allows representations shaped by the generative task to bolster grammatical feature extraction, preserving high diagnostic capability.

3. **End-to-End Interpretability:**
   - In addition to superior correction scores, `DST5-GECC` provides explicit linguistic rationales for its edits: users and downstream systems receive both the error diagnosis and the corrected sentence in a single unified inference pass.

---

## Repository Structure

```text
AO-GEC-Research/
├── Proposed Model Architecture.png                 # Model architecture diagram (Figure 1)
├── ao-gec-using-t5-classifier-baseline.ipynb       # Baseline 1: Single-task error classifier
├── ao-gec-using-t5-generation-baseline.ipynb       # Baseline 2: Single-task Seq2Seq GEC
├── ao-gec-generation-and-classifier-using-t5.ipynb # Proposed DST5-GECC dual-stream model
└── README.md                                       # Project documentation
```

### Notebook Overview

1. **`ao-gec-using-t5-classifier-baseline.ipynb` (Baseline 1)**
   - Standalone diagnostic pipeline using T5 encoder features and a deep classification MLP.
   - Evaluated on test Macro $F_1$ (**90.23%**), Precision (**90.15%**), and Recall (**90.31%**).

2. **`ao-gec-using-t5-generation-baseline.ipynb` (Baseline 2)**
   - Standard sequence-to-sequence baseline mapping uncorrected text to gold targets without error labels.
   - Evaluated on test **GLEU** (**66.27**) and **chrF** (**91.29**).

3. **`ao-gec-generation-and-classifier-using-t5.ipynb` (Proposed `DST5-GECC`)**
   - The primary dual-stream multi-task model implementing the staged training curriculum, class-balanced focal loss, GLEU, and $M^2$ metrics.
   - Achieves the top generative test performance (**68.81 GLEU**, **91.98 chrF**) while maintaining **89.62% Macro-$F_1$** on classification.

---

## End-to-End Pipelined Inference

The trained `DST5-GECC` model performs two-pass inference via `predict_single()`:
1. **Pass 1:** Feeds the uncorrected sentence to Stream A using a neutral prompt to classify the error type.
2. **Pass 2:** Automatically formats an error-conditioned prompt using the predicted label and generates the correction with Stream B using beam search.

```
Raw Input: "Ijaarani jiru isaan mana haaraa."
    │
    ▼ (Stream A: Neutral Diagnostic Pass)
Predicted Error: word-order-error
    │
    ▼ (Stream B: Conditioned Generation Pass)
Corrected Output: "isaan mana haaraa Ijaarani jiru."
```

### Real Demo Inference Samples

| Input Erroneous Sentence | Predicted Error (Stream A) | Model Correction (Stream B) |
| :--- | :--- | :--- |
| `Abubakir fi Galgaloon bara kana barataa cimoo jedhamuu badhaafame turre .` | `subject-verb-agreement-error` | `Abubakir fi Galgaloon bara kana barattoota cimoo jedhamuun badhaafamanii turan.` |
| `Ijaarani jiru isaan mana haaraa.` | `word-order-error` | `isaan mana haaraa Ijaarani jiru.` |
| `Isheen kolfanii hojii ishee xumurte.` | `adverb-verb-agreement-error` | `isheen kolfitee hojii ishee xumurte.` |
| `Inistiitiyuutiin Siinqee dubartootaa guyyaa idil addunyaa kabaje .` | `subject-verb-agreement-error` | `Inistiitiyuutiin Siinqee dubartootaa guyyaa idil addunyaan kabaje.` |

---

## Experimental Setup & Quickstart

The experiments are designed to run in GPU-accelerated notebook environments (such as Kaggle with a T4 or A100 GPU, or a local CUDA machine).

### 1. Requirements

```bash
pip install torch transformers datasets accelerate evaluate scikit-learn pandas numpy matplotlib seaborn wandb tqdm
```

### 2. Dataset Setup
The dataset used across all notebooks is hosted on Kaggle:
- Dataset: `solanadh/ao-gec-dataset-with-error-type` (or `solanadh/aogec-dataset-with-error-type-v03`)
- Expected split files:
  - `AOGEC_Dataset_Labeled_train.csv`
  - `AOGEC_Dataset_Labeled_eval.csv`
  - `AOGEC_Dataset_Labeled_test.csv`
- Each split contains three primary columns: `input`, `target`, and `error_type`.

If running locally, update `data_dir` in each notebook's `CFG` class:
```python
@dataclass
class CFG:
    data_dir: str = "./data"  # path containing CSV splits
    model_name: str = "t5-base"
    batch_size: int = 8
    grad_accum_steps: int = 4
    amp_dtype: str = "bf16"   # "bf16" for Ampere/Ada/Hopper, "fp16" for T4/V100
```

### 3. Execution Order
1. Run `ao-gec-using-t5-classifier-baseline.ipynb` to evaluate standalone diagnostic performance (**Baseline 1**).
2. Run `ao-gec-using-t5-generation-baseline.ipynb` to evaluate standalone seq2seq correction (**Baseline 2**).
3. Run `ao-gec-generation-and-classifier-using-t5.ipynb` for the full **`DST5-GECC`** joint multi-task pipeline.

---

## Roadmap & Future Directions

- [ ] **Multilingual Pretrained Encoders:** Benchmark `google/mt5-base` and `google/byt5-base` to evaluate whether subword or byte-level representations capture Cushitic agglutination better than standard `t5-base`.
- [ ] **Multi-Error Sentences:** Extend the taxonomy and conditioning framework to sentences containing multiple co-occurring error types.
- [ ] **Morphology-Guided Data Augmentation:** Build rule-based linguistic perturbation scripts for Afaan Oromoo to synthetically augment rare error configurations.

---

## Citation & Acknowledgments

If you find this research or dataset helpful in your work, please cite:

```bibtex
@misc{solanadh2026dst5gecc,
  author = {Solana Dh.},
  title = {DST5-GECC: Dual-Stream T5 for Afaan Oromoo Grammatical Error Correction and Classification},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/solanadh/AO-GEC-Research}}
}
```

*Special thanks to the open-source NLP and Ethiopian language computing communities working to advance low-resource language technologies.*
