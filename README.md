# DST5-GECC: Dual-Stream T5 for Afaan Oromoo Grammatical Error Correction & Classification

[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/solanadh/ao-gec-dataset-with-error-type)
[![Model](https://img.shields.io/badge/PyTorch-T5_MultiTask-EE4C2C?logo=pytorch)](https://github.com/solanadh/AO-GEC-Research)

Official implementation of **`DST5-GECC`** (**D**ual-**S**tream **T5** for **G**rammatical **E**rror **C**orrection and **C**lassification), a multi-task framework for Afaan Oromoo.

Instead of relying on black-box sequence translation or disconnected sequential pipelines, `DST5-GECC` splits representations from a shared T5 encoder into two simultaneous streams: **Stream A** identifies the underlying grammatical error category, while **Stream B** conditions generation on the error type to produce targeted, surgical corrections.

<p align="center">
  <img src="Proposed%20Model%20Architecture.png" alt="DST5-GECC Architecture" width="80%"/>
</p>

---

## Model Overview

The proposed architecture integrates diagnosis and correction into a unified model:

- **Shared Encoder:** A single T5 encoder processes the input text, building a shared representation space that informs both diagnostic and generative heads.
- **Stream A (Diagnostic Classification):** Takes unannotated text, pools encoder representations across tokens, and classifies the sentence into one of four grammatical error categories.
- **Stream B (Conditioned Correction):** Generates corrected text by conditioning the T5 decoder on the diagnosed error type, focusing edits on the actual error locus.
- **Two-Pass Inference Engine:** At test time, the model first performs global error classification across the batch, then dynamically structures label-conditioned prompts to decode all corrected sentences in a single parallelized pass.
- **Staged Training Curriculum:** Training progresses through three stages: classifier warm-up with a frozen decoder, joint multi-task training with class-balanced focal loss, and late-stage generation refinement.

---

## Dataset

- **Dataset Link:** [AO_GEC_Dataset_With_Error_Type](https://www.kaggle.com/datasets/solanadh/ao-gec-dataset-with-error-type)
- **Overview:** A parallel corpus of approximately 16,000 Afaan Oromoo sentence pairs gathered from social media, educational materials, and native-speaker drafts.
- **Error Taxonomies:** Focuses on structural grammar across four major error categories:
  - `subject-verb-agreement-error`: Mismatches in person, number, or gender inflections.
  - `noun-adjective-agreement-error`: Disagreements between head nouns and their modifiers.
  - `adverb-verb-agreement-error`: Incompatible adverbial modifiers and verb tense/aspect.
  - `word-order-error`: Disruptions to standard canonical SOV word ordering.

---

## Experimental Results

Test set performance compared against single-task baselines:

| Model Architecture | Classification Macro-$F_1$ | Generation chrF | Generation GLEU |
| :--- | :---: | :---: | :---: |
| Baseline 1 *(Classification Only)* | **90.23%** | — | — |
| Baseline 2 *(Generation Only)* | — | 91.29 | 66.27 |
| **`DST5-GECC`** *(Proposed Multi-Task)* | 89.62% | **91.98** | **68.81** |

- **Enhanced Correction:** Conditioning generation on the error label yields higher correction quality (**68.81 GLEU** vs. 66.27 for standard seq2seq), reducing spurious paraphrasing.
- **Preserved Classification:** Multi-task training retains **89.62% Macro-$F_1$**, staying within 0.61% of the dedicated single-task classifier while providing simultaneous generative correction.

---

## Codebase Structure

The repository is organized into self-contained Jupyter notebooks:

- **`ao-gec-generation-and-classifier-using-t5.ipynb`**: Implementation of the full `DST5-GECC` model, including multi-task loss handling, the three-stage training curriculum, and two-pass batched inference.
- **`ao-gec-using-t5-classifier-baseline.ipynb`**: Implementation and evaluation of the standalone error classification baseline (Baseline 1).
- **`ao-gec-using-t5-generation-baseline.ipynb`**: Implementation and evaluation of the standard sequence-to-sequence GEC baseline (Baseline 2).

---

## Quickstart

### 1. Requirements

```bash
pip install torch transformers datasets accelerate evaluate scikit-learn pandas numpy tqdm
```

### 2. Running the Experiments

The notebooks are self-contained and pre-configured for GPU execution (Kaggle T4/A100 or local CUDA):
1. Open and run `ao-gec-generation-and-classifier-using-t5.ipynb` to train and evaluate the proposed multi-task model.
2. Run `ao-gec-using-t5-classifier-baseline.ipynb` and `ao-gec-using-t5-generation-baseline.ipynb` to reproduce the single-task baselines.
