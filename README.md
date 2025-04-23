# CHNCI: Chinese Cyberbullying Incident Detection Dataset

This repository contains the **CHNCI dataset** and associated code for the paper:

---

## 🧠 Overview

Cyberbullying often emerges around real-world incidents, rapidly escalating online and causing serious harm. Traditional hate speech detection methods fail to capture the **incident-driven**, **context-dependent**, and **personalized** nature of cyberbullying.

We propose:
- A novel **human-machine collaborative annotation framework**.
- The first **Chinese Cyberbullying Incident Detection dataset (CHNCI)**.
- A set of strong **explanation-based detection baselines**.
- Evaluation criteria for **incident-level cyberbullying prediction**.

---

## 📦 Dataset: CHNCI

- **Size**: 220,676 comments  
- **Incidents**: 91 real-world trending topics  
- **Platforms**: Weibo, Douyin, Xiaohongshu, Bilibili  
- **Genres**: Business, Entertainment, Sports, Society, Politics  
- **Annotations**: Human-verified, based on LLM-generated explanations  
- **Cyberbullying Ratio**: ~19%  

### Structure:
Each instance contains:
- `timestamp`: posting time
- `platform`: data source
- `content`: comment text
- `label`: 1 (cyberbullying) or 0 (non-cyberbullying)

---

## ⚙️ Methods

We introduce an ensemble method combining three LLM-based strategies:

- **Paraphraser-based**
- **Chain-of-Thought (CoT)-based**
- **Multi-Agent voting strategy**

All methods generate **explanations** to assist annotators and improve interpretability.

---





