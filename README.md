# DST5-GECC: A Dual-Stream T5-Based Multi-Task Framework for Joint Grammatical Error Correction and Error Type Classification in Low-Resource Afaan Oromo

[![Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/solanadh/ao-gec-dataset-with-error-type)
[![Model](https://img.shields.io/badge/PyTorch-T5_MultiTask-EE4C2C?logo=pytorch)](https://github.com/Solenabera/Afan-Oromo-Grammatical-Error-Correction-and-Error-Type-Classification)

This is the official repository for DST5-GECC (A Dual-Stream T5-Based Multi-Task Framework for Joint Grammatical Error Correction and Error Type Classification in Low-Resource Afaan Oromo). Unlike traditional black-box models or disjointed multi-step pipelines, DST5-GECC introduces a unified multi-task approach. By routing representations from a shared T5 encoder into two concurrent streams, the architecture simultaneously classifies the specific grammatical error (Stream A) and uses that classification to generate precise, targeted corrections (Stream B).

<p align="center">
  <img src="Proposed%20Model%20Architecture.png" alt="DST5-GECC Architecture" width="80%"/>
  <br>
  <em>Figure 1: Architecture of the proposed DST5-GECC multi-task framework.</em>
</p>

---

## Model Overview

The proposed architecture integrates diagnosis and correction into a unified model:

- **Shared Encoder:** A single T5 encoder processes the input text, building a shared representation space that informs both diagnostic and generative heads.
- **Stream A (Classification Stream):** Takes unannotated text, pools encoder representations across tokens, and classifies the sentence into one of four grammatical error categories.
- **Stream B (Generation Stream):** Generates corrected text by conditioning the T5 decoder on the diagnosed error type, focusing edits on the actual error locus.
- **Two-Pass Inference Engine:** At test time, the model first performs global error classification across the batch, then dynamically structures label-conditioned prompts to decode all corrected sentences in a single parallelized pass.
- **Staged Training Curriculum:** Training progresses through three stages: classifier warm-up with a frozen decoder, joint multi-task training with class-balanced focal loss, and late-stage generation refinement.

---

## Dataset

- **Overview:** A parallel corpus of approximately 16,000 Afaan Oromoo sentence pairs gathered from social media, educational materials, and native-speaker drafts.
- **Dataset Link:** [AO_GEC_Dataset_With_Error_Type](https://www.kaggle.com/datasets/solanadh/ao-gec-dataset-with-error-type)
- **Error Taxonomies:** Focuses on structural grammar across four major error categories:
  - `subject-verb-agreement-error`: Mismatches in person, number, or gender inflections.
  - `noun-adjective-agreement-error`: Disagreements between head nouns and their modifiers.
  - `adverb-verb-agreement-error`: Incompatible adverbial modifiers and verb tense/aspect.
  - `word-order-error`: Disruptions to standard canonical SOV word ordering.

---

## Experimental Results

Test set performance compared against single-task baselines:

| Model Architecture | Classification Macro-F1 | Generation chrF | Generation GLEU |
| :--- | :---: | :---: | :---: |
| Baseline 1 *(Classification Only)* | **90.23%** | — | — |
| Baseline 2 *(Generation Only)* | — | 91.29 | 66.27 |
| **`DST5-GECC`** *(Proposed Multi-Task)* | 89.62% | **91.98** | **68.81** |

- **Enhanced Correction:** Conditioning generation on the error label yields higher correction quality (**68.81 GLEU** vs. 66.27 for standard seq2seq), reducing unnecessary edits.
- **Preserved Classification:** Multi-task training retains **89.62% Macro-F1**, staying within 0.61% of the dedicated single-task classifier while providing simultaneous generative correction.

---

## Codebase Structure

The repository is organized into self-contained Jupyter notebooks:

- **`ao-gec-generation-and-classifier-using-t5.ipynb`**: Implementation of the full `DST5-GECC` model, including multi-task loss handling, the three-stage training curriculum, and two-pass batched inference.
- **`ao-gec-using-t5-classifier-baseline.ipynb`**: Implementation and evaluation of the standalone error classification baseline (Baseline 1).
- **`ao-gec-using-t5-generation-baseline.ipynb`**: Implementation and evaluation of the standard sequence-to-sequence GEC baseline (Baseline 2).

### Code Description (`ao-gec-generation-and-classifier-using-t5.ipynb`)

A precise, general overview of the core components implemented in the main proposed model:

- **Model Class (`T5MultiTask`):** Builds on Hugging Face's `T5ForConditionalGeneration` using a shared encoder that branches into:
  - *Stream A:* A dual-pooling operator (mean + max pooling concatenation) feeding a 4-layer MLP classifier with LayerNorm and GELU.
  - *Stream B:* The autoregressive T5 decoder cross-attending to label-conditioned encoder states.
- **Data Pipeline (`DualPromptGECDataset`):** Pre-tokenizes two prompt representations per instance—a neutral prompt for Stream A diagnosis and an error-conditioned prompt for Stream B sequence correction.
- **Loss Optimization:** Integrates autoregressive cross-entropy (with $\epsilon = 0.1$ label smoothing) and class-weighted focal loss ($\gamma = 1.6$) to mitigate error class imbalances.
- **Curriculum Training Routine:** Manages a three-stage schedule—Stage 0 (decoder frozen for classifier warm-up), Stage 1 (joint multi-task optimization), and Stage 2 (generation refinement with decayed classification weight).
- **Batched Two-Pass Inference (`generate_and_classify`):** Executes efficient evaluation by first predicting error classes globally across the batch, then dynamically constructing conditioned prompts for parallel beam-search decoding.

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