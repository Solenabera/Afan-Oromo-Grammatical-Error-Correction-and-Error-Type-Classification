# DST5-GECC: Dual-Stream T5 for Afaan Oromoo Grammatical Error Correction & Classification

[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/solanadh/ao-gec-dataset-with-error-type)
[![Model](https://img.shields.io/badge/PyTorch-T5_MultiTask-EE4C2C?logo=pytorch)](https://github.com/solanadh/AO-GEC-Research)

This repository provides the official implementation of **`DST5-GECC`** (**D**ual-**S**tream **T5** for **G**rammatical **E**rror **C**orrection and **C**lassification), a unified multi-task framework designed for Afaan Oromoo (Cushitic language, ~60M speakers). 

Instead of treating GEC as an uninterpretable black-box translation or relying on slow, error-prone sequential pipelines (classify $\to$ generate), `DST5-GECC` splits hidden representations from a shared T5 encoder into two simultaneous streams: **Stream A** identifies the fine-grained grammatical error category, while **Stream B** conditions generation on the error type to produce precise, localized corrections.

---

## Model Architecture

<p align="center">
  <img src="Proposed%20Model%20Architecture.png" alt="DST5-GECC Architecture" width="85%"/>
</p>

- **Shared T5 Encoder:** Simultaneously processes two prompt formulations with gradient checkpointing for memory efficiency.
- **Stream A (Diagnostic Classification):** Wraps raw text in a neutral prompt (`"classify the grammatical error in afaan oromo: [Sentence]"`), applies dual pooling (mean + max pooling $\to 2 \times d_{\text{model}}$), and passes representations through a 4-layer MLP with LayerNorm/GELU to classify the error type.
- **Stream B (Conditioned Correction):** Formulates a label-conditioned prompt (`"correct: [Label] in Afaan Oromo: [Sentence]"`), feeding cross-attention states to the autoregressive T5 decoder with label smoothing ($\epsilon = 0.1$).
- **Two-Pass Batched Inference Engine:** 
  1. *Pass 1 (Global Diagnosis):* Stream A predicts discrete error labels across the batch.
  2. *Pass 2 (Batched Generation):* Stream B embeds predicted labels into conditioned prompts and decodes all corrections in a single parallelized batch using beam search (width = 4, length penalty = 0.8).

---

## Dataset

- **Kaggle Link:** [AO_GEC_Dataset_With_Error_Type](https://www.kaggle.com/datasets/solanadh/ao-gec-dataset-with-error-type)
- **Size & Scope:** ~16,000 parallel Afaan Oromoo sentence pairs (average length: 8 words).
- **Sources:** Social media (Facebook, Telegram), digital school textbooks, and native-speaker drafts.
- **Targeted Error Taxonomies (4 structural categories, manually verified by language experts):**
  1. `subject-verb-agreement-error`: Mismatches in person, number, or gender inflections.
  2. `noun-adjective-agreement-error`: Broken nominal suffix alignment between nouns and modifiers.
  3. `adverb-verb-agreement-error`: Incompatible adverbial modifiers and verb tense/aspect.
  4. `word-order-error`: Disruptions to standard canonical SOV word ordering.

---

## Experimental Results

Evaluated on the held-out test split against a dedicated single-task classifier (Baseline 1) and a standard sequence-to-sequence model (Baseline 2):

| Model Architecture | Classification (Stream A)<br>Precision | Classification (Stream A)<br>Recall | Classification (Stream A)<br>Macro-$F_1$ | Generation (Stream B)<br>chrF | Generation (Stream B)<br>GLEU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1** *(Classification Only)* | **90.15%** | **90.31%** | **90.23%** | — | — |
| **Baseline 2** *(Generation Only)* | — | — | — | 91.29 | 66.27 |
| **Proposed `DST5-GECC`** *(Ours)* | 89.32% | 89.93% | 89.62% | **91.98** | **68.81** |

**Key Takeaways:**
- **Higher Correction Accuracy:** Error-conditioning in Stream B yields **+2.54 GLEU** and **+0.69 chrF** over standard seq2seq, suppressing spurious paraphrasing.
- **Preserved Diagnostic Power:** Joint multi-task learning achieves **89.62% Macro-$F_1$**, closely matching the dedicated single-task classifier while adding end-to-end correction.

---

## Repository Structure

```text
AO-GEC-Research/
├── Proposed Model Architecture.png                 # Model architecture diagram
├── ao-gec-using-t5-classifier-baseline.ipynb       # Baseline 1: Single-task error classifier
├── ao-gec-using-t5-generation-baseline.ipynb       # Baseline 2: Single-task Seq2Seq GEC
├── ao-gec-generation-and-classifier-using-t5.ipynb # Proposed DST5-GECC multi-task model
└── README.md
```

---

## Quickstart & Usage

### 1. Requirements

```bash
pip install torch transformers datasets accelerate evaluate scikit-learn pandas numpy tqdm
```

### 2. Execution

Each notebook is self-contained and pre-configured for Kaggle GPU (T4/A100) or local CUDA execution:
1. `ao-gec-using-t5-classifier-baseline.ipynb`: Trains and evaluates the standalone diagnostic classifier.
2. `ao-gec-using-t5-generation-baseline.ipynb`: Trains and benchmarks the standalone seq2seq model.
3. `ao-gec-generation-and-classifier-using-t5.ipynb`: Executes the full `DST5-GECC` dual-stream multi-task training and two-pass inference pipeline.

### 3. Inference Example

```python
# Two-pass batched inference demo
input_text = "Ijaarani jiru isaan mana haaraa."
# Pass 1: Stream A predicts error type -> 'word-order-error'
# Pass 2: Stream B generates correction conditioned on predicted label
# Output: "isaan mana haaraa Ijaarani jiru."
```
