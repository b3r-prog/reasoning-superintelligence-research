# 🌱 Deep Dives: Promising Research Areas

> Detailed breakdowns of the most fertile research directions in reasoning and superintelligence.

---

## 1. Test-Time Compute Scaling

### Overview
The core insight: you can trade inference compute for better answers. Instead of just sampling once, allocate a "thinking budget" to search through reasoning paths and self-verify.

### Current State (2025)
- OpenAI's o1 and o3 demonstrate dramatic gains on math, coding, and scientific reasoning
- DeepSeek-R1 shows pure RL training (without SFT warmup) can produce emergent chain-of-thought reasoning
- Google's Gemini 2.0 Thinking and Anthropic's Claude 3.7 Sonnet extended thinking both implement variants

### Key Techniques
- **Best-of-N sampling** — generate N responses, pick the best via reward model
- **Process reward models (PRMs)** — score each reasoning step rather than just the final answer
- **Monte Carlo Tree Search** — tree-structured search with rollouts
- **Self-consistency** — majority vote over diverse reasoning chains
- **Beam search over reasoning tokens** — sequential decoding with pruning

### Open Questions
1. What's the optimal search algorithm? (MCTS vs beam vs best-of-N)
2. How do we train PRMs without expensive human step annotations?
3. When does more compute help vs. hurt? (overthinking problem)
4. Is there a theoretical ceiling? Can test-time compute substitute for more parameters?

### Key Papers
- [Scaling LLM Test-Time Compute Optimally (2024)](https://arxiv.org/abs/2408.03314)
- [Let's Verify Step by Step (2023)](https://arxiv.org/abs/2305.20050)
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948)
- [Self-Consistency (2022)](https://arxiv.org/abs/2203.11171)

---

## 2. Mechanistic Interpretability

### Overview
The science of reverse-engineering neural network computations. Rather than asking "what does it output?" we ask "how does it compute that?" Goal: understand the algorithms implemented by transformers.

### Current State (2025)
- Sparse autoencoders (SAEs) can decompose residual stream activations into interpretable features
- Circuit-level analysis has explained specific capabilities (IOI task, greater-than, docstring generation)
- Anthropic's model organisms work has found interpretable features at scale in Claude

### Key Techniques
- **Activation patching** — intervene on model activations to trace causal paths
- **Sparse autoencoders (SAEs)** — learn overcomplete dictionaries of monosemantic features
- **Logit attribution** — decompose output logits into contributions from each component
- **Attention pattern analysis** — identify what each head attends to
- **Probing classifiers** — test whether a concept is linearly represented

### Open Questions
1. Does mech interp scale to frontier models? (100B+ parameters)
2. Can we detect intentional deception or sandbagging via activations?
3. What's the relationship between circuits and capabilities?
4. Can interpretability insights guide architecture design?

### Key Papers
- [Toy Models of Superposition (2022)](https://transformer-circuits.pub/2022/toy_model/index.html)
- [Towards Monosemanticity (2023)](https://transformer-circuits.pub/2023/monosemanticity/index.html)
- [Scaling SAEs (2024)](https://arxiv.org/abs/2406.04093)
- [IOI Circuit (2022)](https://arxiv.org/abs/2211.00593)

---

## 3. Formal Reasoning & Neural Theorem Proving

### Overview
Can AI produce proofs that can be machine-verified? Lean 4 and Isabelle provide formal foundations; the question is whether LLMs can generate correct formal mathematics.

### Current State (2025)
- AlphaGeometry 2 achieves silver medal performance on IMO geometry
- AlphaProof passes 4 of 6 IMO 2024 problems
- Lean + LLM hybrid systems can close many olympiad-level problems

### Key Techniques
- **Autoformalization** — translate natural language math to Lean/Isabelle
- **Proof search with LLM guidance** — use LLMs to propose tactics in a formal proof engine
- **Synthetic data generation** — generate formal statements and proofs at scale
- **Curriculum learning** — start with easy lemmas, bootstrap to harder theorems

### Open Questions
1. Can we autoformalize graduate-level mathematics reliably?
2. What's the right granularity of formal proof steps for LLMs?
3. Can LLMs discover genuinely new mathematical theorems?
4. How do we generate diverse formal training data at scale?

### Key Papers
- [AlphaGeometry (2024)](https://www.nature.com/articles/s41586-023-06747-5)
- [Autoformalization (2022)](https://arxiv.org/abs/2205.12615)
- [Hypertree Proof Search (2022)](https://arxiv.org/abs/2205.11491)
- [miniF2F benchmark](https://github.com/openai/miniF2F)

---

## 4. Scalable Oversight

### Overview
How do we supervise AI systems that are smarter than the humans supervising them? This is THE central challenge as AI approaches superintelligence.

### Current State (2025)
- Debate has been tested but remains unscaled
- Weak-to-strong generalization shows strong models can often exceed their supervisors even with weak labels
- IDA (Iterated Distillation and Amplification) remains theoretical

### Key Techniques
- **Debate** — two AI agents argue opposing sides; humans judge
- **Recursive reward modeling** — use AI assistants to help humans evaluate AI
- **Process supervision** — supervise reasoning steps not just final answers
- **Weak-to-strong generalization** — stronger model trained to elicit latent knowledge

### Open Questions
1. Does debate work when both debaters are much stronger than the judge?
2. Can we formalize what "latent knowledge" means in a neural network?
3. How do we prevent colluding debaters?
4. Is there a theoretical limit to scalable oversight without an honest prior?

### Key Papers
- [Scalable Oversight via Debate (2018)](https://arxiv.org/abs/1805.00899)
- [Weak-to-Strong Generalization (2023)](https://arxiv.org/abs/2312.09390)
- [IDA (2018)](https://arxiv.org/abs/1810.08575)
- [ELK (2021)](https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC0/)

---

## 5. Self-Improvement & Bootstrapping

### Overview
Can models improve themselves — generating better training data, reward signals, or even architecture designs — in a virtuous cycle?

### Current State (2025)
- STaR/Quiet-STaR show models can bootstrap reasoning from generated rationales
- Self-play methods (SPIN, self-rewarding LMs) show iterative improvement
- AlphaZero-style self-play for reasoning is an active research direction

### Key Techniques
- **Rationale generation + filtering** — generate rationales, keep ones that lead to correct answers
- **Self-rewarding** — model scores its own outputs and trains on high-scoring ones
- **Constitutional self-critique** — model critiques itself against a principle set
- **Rejection sampling fine-tuning** — generate many solutions, filter, fine-tune

### Open Questions
1. When does self-improvement lead to capability gain vs. reward hacking?
2. Can self-improvement compound exponentially or does it plateau?
3. How do we maintain diversity in self-play to avoid mode collapse?
4. What are the safety boundaries for self-modifying systems?

### Key Papers
- [STaR (2022)](https://arxiv.org/abs/2203.14465)
- [Quiet-STaR (2024)](https://arxiv.org/abs/2403.09629)
- [SPIN (2024)](https://arxiv.org/abs/2401.01335)
- [Self-Rewarding Language Models (2024)](https://arxiv.org/abs/2401.10020)

---

*Each research area has a corresponding issue label in this repo. Open an issue to discuss any of these topics.*
