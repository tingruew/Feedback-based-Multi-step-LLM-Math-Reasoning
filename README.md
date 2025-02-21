# Feedback-based Multi-step LLM Reasoning in Math

A survey of multi-step reasoning in large language models for math.

![](images/taxonomy.png)

Detailed information can be found in our [survey paper](https://arxiv.org/pdf/2502.14333).

```Bibtex
@article{wei2025survey,
  title={A Survey on Feedback-based Multi-step Reasoning for Large Language Models on Mathematics},
  author={Wei, Ting-Ruen and Liu, Haowei and Wu, Xuyang and Fang, Yi},
  journal={arXiv preprint arXiv:2502.14333},
  year={2025}
}
```
## Table of Contents
- [Background in LLM Multi-step Reasoning](#background-in-llm-multi-step-reasoning)
- [Papers](#papers)
  - [Step-level Feedback](#step-level-feedback)
    - [Aggregation](#aggregation)

---

### Background in LLM Multi-step Reasoning
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf)
- [Solving Math Word Problems with Process and Outcome-based Feedback](https://arxiv.org/pdf/2211.14275)
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/pdf/2110.14168)
- [Let's Verify Step by Step](https://openreview.net/pdf?id=v8L0pN6EOi)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/pdf/2203.11171)
- [A Survey of Monte Carlo Tree Search Methods](https://ieeexplore.ieee.org/document/6145622)

---

## Papers

### Training-based

#### Step-level Feedback

- #### **Aggregation**
  
|Title | Venue | Main Image | Paper|
| :--- | :--: | :---: | :---: |
|Making Large Language Models Better Reasoners with Step-Aware Verifier| ACL 2023 | <img width="1200" alt="image" src="images/aggregation/making.png"> | [Link](https://aclanthology.org/2023.acl-long.291.pdf)|
|Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations| ACL 2024 | <img width="1200" alt="image" src="images/aggregation/math-shepherd.png"> | [Link](https://aclanthology.org/2024.acl-long.510.pdf)|
|Evaluating Mathematical Reasoning Beyond Accuracy| AAAI 2025 | <img width="1200" alt="image" src="images/aggregation/evaluating.png"> | [Link](https://arxiv.org/pdf/2404.05692)|
|SELF-EXPLORE: Enhancing Mathematical Reasoning in Language Models with Fine-grained Rewards| EMNLP 2024 | <img width="1200" alt="image" src="images/aggregation/self-explore.png"> | [Link](https://aclanthology.org/2024.findings-emnlp.78.pdf)|
|VerifierQ: Enhancing LLM Test Time Compute with Q-Learning-based Verifiers| Arxiv 2024 | <img width="1200" alt="image" src="images/aggregation/verifierq.png"> | [Link](https://arxiv.org/pdf/2410.08048)|
|AutoPSV: Automated Process-Supervised Verifier| NeurIPS 2024 | <img width="1200" alt="image" src="images/aggregation/autopsv.png"> | [Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/9246aa822579d9b29a140ecdac36ad60-Paper-Conference.pdf)|
|Multi-step Problem Solving Through a Verifier: An Empirical Analysis on Model-induced Process Supervision| EMNLP 2024 | <img width="1200" alt="image" src="images/aggregation/multi-step.png"> | [Link](https://aclanthology.org/2024.findings-emnlp.429.pdf)|
|The Reason behind Good or Bad: Towards a Better Mathematical Verifier with Natural Language Feedback| Arxiv 2024 | <img width="1200" alt="image" src="images/aggregation/the.png"> | [Link](https://arxiv.org/pdf/2406.14024v1)|
|Advancing Process Verification for Large Language Models via Tree-Based Preference Learning| EMNLP 2024 | <img width="1200" alt="image" src="images/aggregation/advancing.png"> | [Link](https://aclanthology.org/2024.emnlp-main.125.pdf)|
|Improve Mathematical Reasoning in Language Models by Automated Process Supervision| Arxiv 2024 | <img width="1200" alt="image" src="images/aggregation/improve.png"> | [Link](https://arxiv.org/pdf/2406.06592)|
|Free Process Rewards without Process Labels| Arxiv 2024 | <img width="1200" alt="image" src="images/aggregation/free.png"> | [Link](https://arxiv.org/pdf/2412.01981)|
|The Lessons of Developing Process Reward Models in Mathematical Reasoning| Arxiv 2025 | <img width="1200" alt="image" src="images/aggregation/lessons.png"> | [Link](https://arxiv.org/pdf/2501.07301)|
|What Are Step-Level Reward Models Rewarding? Counterintuitive Findings from MCTS-Boosted Mathematical Reasoning| AAAI 2025 | <img width="1200" alt="image" src="images/aggregation/what.png"> | [Link](https://arxiv.org/pdf/2412.15904)|
|Process Reward Model with Q-Value Rankings| ICLR 2025 | <img width="1200" alt="image" src="images/aggregation/process.png"> | [Link](https://arxiv.org/pdf/2410.11287)|
|Entropy-Regularized Process Reward Model| Arxiv 2024 | <img width="1200" alt="image" src="images/aggregation/entropy-regularized.png"> | [Link](https://arxiv.org/pdf/2412.11006)|

- #### **Search**
  
|Title | Venue | Main Image | Paper|
| :--- | :--: | :---: | :---: |
|Deductive Beam Search: Decoding Deducible Rationale for Chain-of-Thought Reasoning| COLM 2024 | <img width="1200" alt="image" src="images/search/deductive.png"> | [Link](https://openreview.net/pdf?id=S1XnUsqwr7)|
|MindStar: Enhancing Math Reasoning in Pre-trained LLMs at Inference Time| Arxiv 2024 | <img width="1200" alt="image" src="images/search/mindstar.png"> | [Link](https://arxiv.org/pdf/2405.16265)|
|No Train Still Gain. Unleash Mathematical Reasoning of Large Language Models with Monte Carlo Tree Search Guided by Energy Function| COLM 2024 | <img width="1200" alt="image" src="images/search/no.png"> | [Link](https://openreview.net/pdf?id=S1XnUsqwr7)|
|Let's Reward Step by Step: Step-Level Reward Model as the Navigators for Reasoning| Arxiv 2023 | <img width="1200" alt="image" src="images/search/lets.png"> | [Link](https://arxiv.org/pdf/2310.10080)|
|Token-Supervised Value Models for Enhancing Mathematical Problem-Solving Capabilities of Large Language Models| ICLR 2025 | <img width="1200" alt="image" src="images/search/token-supervised.png"> | [Link](https://openreview.net/pdf?id=6HcnC3pPkp)|
|GRACE: Discriminator-Guided Chain-of-Thought Reasoning| EMNLP 2023 | <img width="1200" alt="image" src="images/search/grace.png"> | [Link](https://aclanthology.org/2023.findings-emnlp.1022.pdf)|
|Offline Reinforcement Learning for LLM Multi-Step Reasoning| Arxiv 2024 | <img width="1200" alt="image" src="images/search/offline.png"> | [Link](https://arxiv.org/pdf/2412.16145)|
|Reward-Guided Speculative Decoding for Efficient LLM Reasoning| Arxiv 2025 | <img width="1200" alt="image" src="images/search/reward-guided.png"> | [Link](https://arxiv.org/pdf/2501.19324)|
|Ensembling Large Language Models with Process Reward-Guided Tree Search for Better Complex Reasoning| Arxiv 2024 | <img width="1200" alt="image" src="images/search/ensembling.png"> | [Link](https://arxiv.org/pdf/2412.15797)|
|AlphaMath Almost Zero: Process Supervision without Process| NeurIPS 2024 | <img width="1200" alt="image" src="images/search/alphamath.png"> | [Link](https://arxiv.org/pdf/2405.03553)|
|Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters| Arxiv 2024 | <img width="1200" alt="image" src="images/search/scaling.png"> | [Link](https://arxiv.org/pdf/2408.03314)|
