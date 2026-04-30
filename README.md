# AIC-199

AIC-199: AI Identification Criteria Dataset. A labeled benchmark of 199 open-source software applications for AI/Classical software discrimination research.

## Overview

AIC-199 is a curated, feature-extracted dataset designed to support research on the automated detection of AI algorithmic elements in application source code — that is, the task of determining whether a software system implements an AI-based computational approach (machine learning, deep learning, generative AI, reinforcement learning) or a classical, rule-based, deterministic algorithm.

The dataset was constructed and used as the experimental basis for the paper:

> A. Ionica, M. Leba, *"From House of Quality to Neural Architecture: Quality-Informed Neural Networks for Interpretable Classification, with an EU AI Act Compliance Application"*, submitted to MDPI Systems, 2026.

It is released as a standalone research resource to support reproducibility and to serve as a benchmark for future work on software classification, AI governance tooling, and EU AI Act compliance instrumentation.

---

## Dataset at a Glance

| Property | Value |
|---|---|
| Total applications | 199 |
| AI class (label = AI) | 100 |
| Classical class (label = Non-AI) | 99 |
| Input features per application | 27 (x₁ – x₂₇) |
| Feature groups | 6 (Dependency, Code Pattern, Artifact, Documentation, Structural, Learning Signal) |
| Application domains covered | 29 |
| Source | Static analysis of public GitHub repositories |
| Format | Microsoft Excel (.xlsx), 3 sheets |

---

## Class Definitions

**AI system** — software that learns statistical or parametric models from data. Includes deep learning frameworks, large language model systems, generative AI tools, computer vision models, reinforcement learning environments, and classical machine learning systems (collaborative filtering, matrix factorisation, statistical NLP) that train from data.

**Classical system** — software whose decision logic is entirely explicit, rule-based, or algorithmically specified at development time, with no learned parameters. Includes deterministic algorithms, rule-based NLP, analytics platforms, DevOps infrastructure, workflow automation tools, and MLOps management tools that operate on AI artefacts without performing learning themselves.

This taxonomy is aligned with the definition of an AI system under **Article 3(1) of Regulation (EU) 2024/1689** (EU AI Act) and the European Commission's Guidelines of 6 February 2025.

---

## Application Domains

The 199 applications span 29 domain categories, covering the full spectrum from clearly AI-native to clearly classical.

**AI domains (18 categories):** LLM & Chatbot, AI Agents & Automation, RAG & Knowledge Base, Computer Vision & Detection, Image/Video/Audio Generation, Code Generation & Assistance, ML & DL Frameworks, NLP & Text, Robotics & Embedded AI, AI Infrastructure, Medical & Bioinformatics, Security & Anomaly Detection, Vector Databases, Recommendation Systems (Classical ML), and others.

**Classical domains (11 categories):** Analytics & BI, Workflow Automation & ETL, DevOps & Infrastructure, DevTools & Static Analysis, Search & Indexing, Classical NLP, Robotics & Control (Classical), Image & Media Processing, AI Infrastructure (Annotation & MLOps tools), and others.

---

## Feature Description

All 27 features are extracted through static analysis of repository source files using Python `ast`, `pathlib`, and `regex` — no code execution is required.

| Group | Features | Type | Description |
|---|---|---|---|
| A — Dependency | x₁ – x₅ | ratio / binary | AI/ML library linkage, LLM SDK, VectorDB, CV/Speech, RL/Sim flags |
| B — Code Pattern | x₆ – x₁₁ | log-ratio | fit/train, predict/infer, generate/chat, embed/encode, reward/step, score/rec call counts |
| C — Artifact | x₁₂ – x₁₅ | binary | Model weight files, tokenizer files, training configs/checkpoints, model cards |
| D — Documentation | x₁₆ – x₁₉ | density / binary | AI keyword density, LLM/agent terminology, benchmark sections, dataset references |
| E — Structural | x₂₀ – x₂₅ | binary / ratio | AI folder structure, GPU/CUDA config, Dockerfile AI services, AI package ratio, CI/CD ML pipeline, AI license |
| F — Learning Signal | x₂₆ – x₂₇ | binary | `trains_model` (training loop code detected), `classical_model_artefact` (serialised classical ML model files) |

Features x₂₆ and x₂₇ (Group F) are new additions introduced to correctly classify classical ML systems (collaborative filtering, statistical NLP) that train from data without deep learning infrastructure.

---

## File Structure
AIC-199_Dataset.xlsx
├── Feature Collection   — 199 rows × 64 columns: app metadata, raw feature values,
│                          normalised features (x₁–x₂₇), QFD flat score, and QINN output
├── Features             — Feature changelog: definitions, revision notes for x₁₁, x₁₂,
│                          x₁₆, x₁₉, and new features x₂₆–x₂₇
└── Weights & Score Summary — 27-feature weight table (wᵢ), expected score distribution
at θ = 0.15, and classification boundary analysis

---

## Usage Notes

- All normalised features (x₁–x₂₇) are ready for direct input to classifiers. Raw counts are provided for reproducibility of the normalisation step.
- The QFD flat score column enables direct comparison with the non-learned baseline reported in the paper.
- The `GitHub URL` column contains the source repository for each application; note that repository contents may have changed since feature extraction.
- Seven applications lie in a genuine borderline zone (QFD score 0.10–0.30) and represent contested cases at the AI/Classical boundary — these are documented in the Score Summary sheet and are of particular interest for boundary analysis research.

---

## Citation

If you use this dataset, please cite:

```bibtex
@article{IonicaLeba2026qinn,
  author  = {Ionica, Andreea; Leba, Monica},
  title   = {From House of Quality to Neural Architecture:
            Quality-Informed Neural Networks for Interpretable Classification, 
            with an EU AI Act Compliance Application},
  journal = {MDPI Systems},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. You are free to share and adapt the material for any purpose, provided appropriate credit is given.

---

## Acknowledgement

This dataset was produced within the framework of the project *"Holistic Cybersecurity Sandbox for Testing Artificial Intelligence Solutions"*, ID PCIDIF/159/PCIDIF_P1/OP1/RSO1.1/PCIDIF_A.
