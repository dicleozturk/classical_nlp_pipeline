# Classical NLP Pipeline (2016–2019)

This repository presents a **modular, production-oriented Natural Language Processing (NLP) pipeline**
developed between **2017 and 2018**, primarily based on **classical machine learning methods**
and **scikit-learn**.

It is organized into modular source code and a dedicated
research_reports section that documents applied research and evaluation work.

The project reflects how **robust, extensible, and maintainable NLP systems** were designed
and deployed **before the widespread adoption of large neural and transformer-based models**,
with a strong emphasis on clarity, modularity, and real-world applicability.

---

## Motivation

Prior to the dominance of deep learning and large language models, many enterprise NLP systems
relied on:
- careful text preprocessing,
- feature-based representations,
- classical supervised and unsupervised learning algorithms,
- and well-structured pipeline architectures.

This repository serves as a **reference implementation** of such systems, highlighting
approaches that remain valuable today for:
- explainability,
- low-resource and constrained environments,
- maintainability and debuggability,
- and educational purposes.

---

## Overview

The pipeline is designed as a **generic and domain-agnostic text processing framework**.
It clearly separates concerns across the NLP workflow, including:
- preprocessing,
- feature extraction,
- modeling,
- evaluation,
- and orchestration.

At its core, the project includes a reusable **text categorization package** that can be
adapted to different datasets and use cases with minimal configuration changes.

---

## Key Characteristics

- Classical NLP and machine learning methods (non-neural)
- scikit-learn–based modeling
- Modular and extensible architecture
- Generic and reusable text categorization components
- Designed with enterprise-scale applications in mind
- Emphasis on clarity, maintainability, and interpretability

---

## Supported NLP Tasks

Depending on configuration and use case, the pipeline supports:
- Text categorization and multi-class classification
- Feature-based text representation
- Keyword and keyphrase extraction
- Topic-oriented analysis
- Rule-based and statistical preprocessing
- Exploratory evaluation and error analysis

---

## High-Level Architecture

Raw Text
↓
Preprocessing
↓
Feature Extraction
↓
Classical ML Models (scikit-learn)
↓
Evaluation & Analysis

Each stage is implemented as an independent module, enabling experimentation,
replacement, and extension without affecting the overall system design.

---

## Research and Experimentation

The repository includes a collection of **research reports and experimental materials**
documenting:
- design decisions,
- methodological trade-offs,
- evaluation observations,
- and lessons learned from applied use cases.

These documents reflect an applied research approach, focusing not only on *how*
components were implemented, but also *why* certain techniques were preferred
in real-world settings.

---

## Reports and Research Documentation

In addition to the source code, this repository includes a set of **research reports**
located under the `research_reports/` directory.

These documents were produced during the development of the pipeline and provide:
- detailed analysis of problem formulations,
- comparisons between different classical approaches,
- evaluation results and observations,
- design rationales and trade-offs,
- reflections on real-world deployment constraints.

The reports complement the codebase by documenting the **analytical and decision-making
processes** behind the system, offering insight into how classical NLP solutions were
designed, assessed, and refined in practice.

---

## Intended Use


It is **not positioned as a state-of-the-art neural NLP system**, but rather as a
carefully designed and well-documented classical alternative.

---

## Disclaimer

This repository is a **generalized and anonymized representation** of an
enterprise-scale NLP pipeline.

It does not contain proprietary code, client-specific logic, sensitive data,
or confidential information. All examples and configurations are provided
for demonstration and educational purposes only.

---

## Author

Developed by **Dicle Öztürk**  
dicle@lucitext.io
