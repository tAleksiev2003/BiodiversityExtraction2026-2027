# Biodiversity Extraction Benchmark — Dev-100 Leaderboard

All models evaluated identically on the 100 validated Dev papers (`data/gold/dev_100_ids.txt`) using the official set-based exact-match evaluator (`src/evaluation/evaluator.py`).

| Rank | Approach | Type | C1 | C1+C2b (Macro) | Micro-F1 |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | **RAG-Anchored Sections (`deepseek-v4.1-flash-1017`, PyMuPDF, kNN k=2)** | LLM (RAG + filtered sections) | **0.5798** | 0.4273 | 0.4286 |
| **2** | **Cross-Architecture Voting Ensemble (4 systems)** | Hybrid Ensemble (LLM + ML) | **0.5785** | **0.4488** | 0.4300 |
| **3** | **Monolithic Full-Text + Guidelines (`Qwen 3.6-27B` / `hu-reasoning`)** | LLM (full text) | **0.5754** | 0.4149 | **0.4515** |
| **4** | **Parallel 4-Chunk + Guidelines + Term-Enum (`deepseek-v4.1-flash-1017`)** | LLM (chunked + quote gate) | 0.5622 | 0.4244 | 0.4163 |
| **5** | **Parallel 4-Chunk + Guidelines + Term-Enum (`Qwen 3.6-27B`)** | LLM (chunked + quote gate) | 0.5593 | 0.4144 | 0.3923 |
| **6** | **Classical Approval-Voting Ensemble (9 Base Models)** | classical ML ensemble | 0.5260 | 0.3783 | 0.3966 |
| **7** | **Monolithic Full-Text + Guidelines (`mistral-large-4-1t-a52b`)** | LLM (full text) | 0.5143 | 0.3751 | 0.3834 |
| **8** | **TF-IDF + Linear SVM** | classical ML | 0.4897 | 0.3675 | 0.3753 |
| **9** | **Parallel 4-Chunk + Translate-Hint (`Qwen 3.6-27B`)** | LLM (chunked + quote gate) | 0.4833 | 0.3456 | 0.3686 |
| **10** | **BoW + Complement Naive Bayes** | classical ML | 0.4799 | 0.3140 | 0.3564 |
| **11** | **k-NN Paper Retrieval (k = 5)** | classical ML / retrieval | 0.4793 | 0.3419 | 0.3610 |
| **12** | **Parallel 4-Chunk + Field Guidelines (`mistral-large-4-1t-a52b`)** | LLM (chunked + quote gate) | 0.4706 | 0.3392 | 0.3919 |
| **13** | **TF-IDF + Logistic Regression** | classical ML | 0.4677 | 0.3657 | 0.3694 |
| **14** | **Parallel 4-Chunk + Field Guidelines (`Qwen 3.6-27B`)** | LLM (chunked + quote gate) | 0.4565 | 0.3339 | 0.3566 |
| **15** | **Sections Pipeline B — English-First + Mapping (`Qwen 3.6-27B`)** | LLM (sections + EN→DE map) | 0.4476 | 0.3090 | 0.3059 |
| **16** | **Parallel 4-Chunk + Text Cleaning (`Qwen 3.6-27B`)** | LLM (chunked + quote gate) | 0.4463 | 0.3321 | 0.3633 |
| **17** | **Parallel 4-Chunk Baseline (`Qwen 3.6-27B`)** | LLM (chunked + quote gate) | 0.4318 | 0.2885 | 0.3152 |
| **18** | **Sections Pipeline B — English-First + Mapping (`glm-5.3-flash`)** | LLM (sections + EN→DE map) | 0.4285 | 0.2920 | 0.2989 |
| **19** | **Parallel 4-Chunk Baseline (`mistral-large-4-1t-a52b`)** | LLM (chunked + quote gate) | 0.4056 | 0.2740 | 0.3112 |
| **20** | **Sections Pipeline B — English-First + Mapping (`deepseek-v4.1-flash-1017`)** | LLM (sections + EN→DE map) | 0.3887 | 0.2685 | 0.2797 |
| **21** | **Dictionary / Keyword Lookup** | rule-based NLP | 0.3438 | 0.2721 | 0.3216 |
| **22** | **Linear-Chain CRF (Viterbi)** | sequence-labelling NLP | 0.3399 | 0.2727 | 0.3110 |
| **23** | **Sections Pipeline B — Direct German, No Translation (`Qwen 3.6-27B`)** | LLM (sections, DE direct) | 0.2498 | 0.2210 | 0.2191 |
| **24** | **Majority Class** | trivial baseline | 0.2055 | 0.1429 | 0.1403 |
| **25** | **Sections Pipeline A — Sequential State-Fold (`Qwen 3.6-27B`)** | LLM (sequential state fold) | 0.1629 | 0.1035 | 0.1360 |
| **26** | **Sections Pipeline B — English-First (`mistral-large-4-1t-a52b`)** | LLM (sections + EN→DE map) | 0.1490 | 0.0929 | 0.1181 |
| **27** | **Random** | trivial baseline | 0.1006 | 0.0607 | 0.0600 |
