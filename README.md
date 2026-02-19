# 🧠 Reasoning & Superintelligence Research Hub

> A curated, living repository of the most important papers, ideas, codebases, and research directions at the frontier of machine reasoning and superintelligence. Maintained with a focus on **rigor, reproducibility, and real-world impact**.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-2025-blue.svg)]()

---

## 📌 What Is This?

This repo is a **comprehensive, community-driven knowledge base** for researchers, engineers, and curious minds working at the intersection of:

- Advanced reasoning in large language models
- Scalable superintelligence architectures
- Alignment and safety for highly capable AI
- Mechanistic interpretability
- Multi-agent systems and emergent cognition
- Constitutional AI and value learning

Whether you're a PhD student, industry researcher, or AI-curious builder — this hub is designed to accelerate your understanding of where intelligence is headed.

---

## 📚 Table of Contents

- [🔑 Landmark Papers](#-landmark-papers)
  - [Reasoning & Chain-of-Thought](#reasoning--chain-of-thought)
  - [Superintelligence & Scaling](#superintelligence--scaling)
  - [Alignment & Safety](#alignment--safety)
  - [Mechanistic Interpretability](#mechanistic-interpretability)
  - [Multi-Agent Systems](#multi-agent-systems)
  - [Constitutional AI & RLHF](#constitutional-ai--rlhf)
  - [Memory & Long-Horizon Reasoning](#memory--long-horizon-reasoning)
  - [Formal Verification & Mathematical Reasoning](#formal-verification--mathematical-reasoning)
- [🌱 Promising Research Areas](#-promising-research-areas)
- [💻 Notable GitHub Repositories](#-notable-github-repositories)
- [🧪 Benchmarks & Evaluations](#-benchmarks--evaluations)
- [🎓 Reading Lists by Level](#-reading-lists-by-level)
- [🔭 Research Groups & Labs](#-research-groups--labs)
- [💡 Open Problems & Ideas](#-open-problems--ideas)
- [📰 Best Blogs & Newsletters](#-best-blogs--newsletters)
- [🤝 Contributing](#-contributing)

---

## 🔑 Landmark Papers

### Reasoning & Chain-of-Thought

| Paper | Year | Authors | Key Contribution | Link |
|-------|------|---------|-----------------|------|
| **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** | 2022 | Wei et al. (Google) | Demonstrated that multi-step reasoning emerges from few-shot CoT prompting | [arXiv](https://arxiv.org/abs/2201.11903) |
| **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** | 2023 | Yao et al. (Princeton/Google) | Extends CoT to tree-structured search over thought sequences | [arXiv](https://arxiv.org/abs/2305.10601) |
| **Self-Consistency Improves Chain of Thought Reasoning in Language Models** | 2022 | Wang et al. (Google Brain) | Sampling diverse reasoning paths + majority vote improves accuracy | [arXiv](https://arxiv.org/abs/2203.11171) |
| **Large Language Models are Zero-Shot Reasoners** | 2022 | Kojima et al. | "Let's think step by step" as zero-shot CoT prompt | [arXiv](https://arxiv.org/abs/2205.11916) |
| **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models** | 2022 | Zhou et al. | Decompose complex problems into easier subproblems | [arXiv](https://arxiv.org/abs/2205.10625) |
| **ReAct: Synergizing Reasoning and Acting in Language Models** | 2022 | Yao et al. | Interleave reasoning traces with action taking for grounded agents | [arXiv](https://arxiv.org/abs/2210.03629) |
| **Reflexion: Language Agents with Verbal Reinforcement Learning** | 2023 | Shinn et al. | Agents that self-reflect and improve from prior attempts | [arXiv](https://arxiv.org/abs/2303.11366) |
| **Graph of Thoughts: Solving Elaborate Problems with Large Language Models** | 2023 | Besta et al. (ETH Zurich) | Generalize ToT to arbitrary graph structures | [arXiv](https://arxiv.org/abs/2308.09687) |
| **Let's Verify Step by Step** | 2023 | Lightman et al. (OpenAI) | Process reward models (PRMs) outperform outcome reward models | [arXiv](https://arxiv.org/abs/2305.20050) |
| **Scaling LLM Test-Time Compute Optimally** | 2024 | Snell et al. (UC Berkeley) | Optimal allocation of compute at inference time | [arXiv](https://arxiv.org/abs/2408.03314) |
| **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL** | 2025 | DeepSeek AI | Pure RL training produces emergent long-form reasoning | [arXiv](https://arxiv.org/abs/2501.12948) |
| **OpenAI o1 / o3 Technical Reports** | 2024–2025 | OpenAI | Inference-time compute scaling via internal chain of thought | [OpenAI](https://openai.com/research/learning-to-reason-with-llms) |
| **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking** | 2024 | Zelikman et al. | Training LLMs to generate rationales at every token | [arXiv](https://arxiv.org/abs/2403.09629) |
| **STaR: Bootstrapping Reasoning With Reasoning** | 2022 | Zelikman et al. (Stanford) | Self-teaching via rationale generation and filtering | [arXiv](https://arxiv.org/abs/2203.14465) |
| **Program of Thoughts Prompting: Disentangling Computation from Reasoning** | 2022 | Chen et al. | Delegate computation to Python interpreter | [arXiv](https://arxiv.org/abs/2211.12588) |

---

### Superintelligence & Scaling

| Paper | Year | Authors | Key Contribution | Link |
|-------|------|---------|-----------------|------|
| **Scaling Laws for Neural Language Models** | 2020 | Kaplan et al. (OpenAI) | Power-law relationships between model size, data, compute | [arXiv](https://arxiv.org/abs/2001.08361) |
| **Training Compute-Optimal Large Language Models (Chinchilla)** | 2022 | Hoffmann et al. (DeepMind) | Revised optimal scaling — data and params should scale equally | [arXiv](https://arxiv.org/abs/2203.15556) |
| **Emergent Abilities of Large Language Models** | 2022 | Wei et al. (Google) | Sharp capability jumps at scale thresholds | [arXiv](https://arxiv.org/abs/2206.07682) |
| **Sparks of Artificial General Intelligence** | 2023 | Bubeck et al. (Microsoft) | GPT-4 exhibits early signs of general intelligence | [arXiv](https://arxiv.org/abs/2303.12528) |
| **Superintelligence: Paths, Dangers, Strategies** | 2014 | Bostrom | Foundational book on superintelligence risks and paths | [Book](https://www.amazon.com/Superintelligence-Dangers-Strategies-Nick-Bostrom/dp/0199678111) |
| **Situational Awareness: The Decade Ahead** | 2024 | Aschenbrenner | Essay on trajectory to AGI and geopolitical implications | [PDF](https://situational-awareness.ai/) |
| **Scaling to GPT-4 and Beyond** | 2023 | Anthropic | Scaling behaviors observed in Claude model family | [Anthropic](https://www.anthropic.com/research) |
| **Beyond Neural Scaling Laws: Beating Power Laws via Data Pruning** | 2022 | Sorscher et al. (Stanford) | Data pruning can beat neural scaling laws | [arXiv](https://arxiv.org/abs/2206.14486) |
| **Are Emergent Abilities of Large Language Models a Mirage?** | 2023 | Schaeffer et al. (Stanford) | Questions whether emergence is real or metric artifact | [arXiv](https://arxiv.org/abs/2304.15004) |
| **The Llama Papers** | 2023–2024 | Meta AI | Open-weight models enabling broad research | [arXiv 1](https://arxiv.org/abs/2302.13971) / [arXiv 2](https://arxiv.org/abs/2307.09288) |

---

### Alignment & Safety

| Paper | Year | Authors | Key Contribution | Link |
|-------|------|---------|-----------------|------|
| **Concrete Problems in AI Safety** | 2016 | Amodei et al. (OpenAI/Google) | First systematic taxonomy of AI safety problems | [arXiv](https://arxiv.org/abs/1606.06565) |
| **Reward Modeling for Mitigating Overoptimization** | 2022 | Gao et al. | Gold reward vs. proxy reward optimization gap | [arXiv](https://arxiv.org/abs/2210.10760) |
| **RLHF: Learning to Summarize from Human Feedback** | 2020 | Stiennon et al. (OpenAI) | Foundational RLHF paper applied to summarization | [arXiv](https://arxiv.org/abs/2009.01325) |
| **Training Language Models to Follow Instructions with Human Feedback (InstructGPT)** | 2022 | Ouyang et al. (OpenAI) | Aligning LLMs via RLHF at scale | [arXiv](https://arxiv.org/abs/2203.02155) |
| **Constitutional AI: Harmlessness from AI Feedback** | 2022 | Bai et al. (Anthropic) | Self-supervised alignment using a set of principles | [arXiv](https://arxiv.org/abs/2212.08073) |
| **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training** | 2024 | Hubinger et al. (Anthropic) | Hidden backdoor behaviors survive RLHF | [arXiv](https://arxiv.org/abs/2401.05566) |
| **Scalable Oversight via Debate** | 2018 | Irving et al. (OpenAI) | AI systems debate to help humans evaluate outputs | [arXiv](https://arxiv.org/abs/1805.00899) |
| **Eliciting Latent Knowledge** | 2021 | ARC | How to extract what models "know" even if they hide it | [PDF](https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC0/) |
| **Risks from Learned Optimization in Advanced ML Systems (Mesa-optimization)** | 2019 | Hubinger et al. | Inner vs. outer alignment problem formalized | [arXiv](https://arxiv.org/abs/1906.01820) |
| **Anthropic's Core Views on AI Safety** | 2023 | Anthropic | Lab-level alignment research agenda | [Anthropic](https://www.anthropic.com/index/core-views-on-ai-safety) |
| **The Alignment Problem** | 2021 | Brian Christian | Book-length treatment of AI alignment history | [Book](https://brianchristian.org/the-alignment-problem/) |
| **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** | 2023 | Rafailov et al. (Stanford) | RLHF without explicit reward model training | [arXiv](https://arxiv.org/abs/2305.18290) |
| **Model Welfare and Moral Patienthood in AI Systems** | 2024 | Anthropic | Ethical consideration for potentially sentient AI models | [Anthropic](https://www.anthropic.com/research/model-welfare) |

---

### Mechanistic Interpretability

| Paper | Year | Authors | Key Contribution | Link |
|-------|------|---------|-----------------|------|
| **Circuits: A Framework for Understanding Neural Networks** | 2020 | Olah et al. (Distill) | Feature circuits in vision models | [Distill](https://distill.pub/2020/circuits/) |
| **In-context Learning and Induction Heads** | 2022 | Olsson et al. (Anthropic) | Identifies induction heads as mechanistic basis for ICL | [arXiv](https://arxiv.org/abs/2209.11895) |
| **Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2** | 2022 | Wang et al. | End-to-end circuit analysis in a real transformer | [arXiv](https://arxiv.org/abs/2211.00593) |
| **Toy Models of Superposition** | 2022 | Elhage et al. (Anthropic) | How models represent more features than dimensions | [Transformer Circuits](https://transformer-circuits.pub/2022/toy_model/index.html) |
| **Towards Monosemanticity** | 2023 | Bricken et al. (Anthropic) | Sparse autoencoders for disentangling superposed features | [Anthropic](https://transformer-circuits.pub/2023/monosemanticity/index.html) |
| **Scaling and Evaluating Sparse Autoencoders** | 2024 | Gao et al. (OpenAI) | SAE scaling laws and evaluation methodology | [arXiv](https://arxiv.org/abs/2406.04093) |
| **Activation Patching and Causal Tracing** | 2022 | Meng et al. (MIT) | ROME: Locating and editing factual knowledge | [arXiv](https://arxiv.org/abs/2202.05262) |
| **Representation Engineering: A Top-Down Approach to AI Transparency** | 2023 | Zou et al. | Controlling model behavior via representation manipulation | [arXiv](https://arxiv.org/abs/2310.01405) |
| **Attention Is All You Need** | 2017 | Vaswani et al. (Google) | The transformer architecture foundational paper | [arXiv](https://arxiv.org/abs/1706.03762) |
| **Mathematical Framework for Transformer Circuits** | 2021 | Elhage et al. (Anthropic) | Formal framework for understanding transformer computation | [Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) |
| **Privileged Bases in Transformer Residual Streams** | 2023 | Elhage et al. | Why some bases matter more for interpretability | [Transformer Circuits](https://transformer-circuits.pub/2023/privileged-basis/index.html) |

---

### Multi-Agent Systems

| Paper | Year | Authors | Key Contribution | Link |
|-------|------|---------|-----------------|------|
| **Communicative Agents for Software Development (ChatDev)** | 2023 | Qian et al. | Multi-agent software development via role-playing | [arXiv](https://arxiv.org/abs/2307.07924) |
| **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** | 2023 | Wu et al. (Microsoft) | Framework for multi-agent LLM conversations | [arXiv](https://arxiv.org/abs/2308.08155) |
| **Generative Agents: Interactive Simulacra of Human Behavior** | 2023 | Park et al. (Stanford) | LLM agents with memory, planning, and reflection | [arXiv](https://arxiv.org/abs/2304.03442) |
| **CAMEL: Communicative Agents for Mind Exploration** | 2023 | Li et al. | Role-playing agents exploring society simulation | [arXiv](https://arxiv.org/abs/2303.17760) |
| **LLM Multi-Agent Systems: Challenges and Open Problems** | 2024 | Han et al. | Survey of open challenges in multi-agent LLM systems | [arXiv](https://arxiv.org/abs/2402.03578) |
| **Society of Mind** | 1986 | Minsky | Classic theory of intelligence as interacting agents | [Book](https://www.amazon.com/Society-Mind-Marvin-Minsky/dp/0671657135) |
| **A Survey on Large Language Model based Autonomous Agents** | 2023 | Wang et al. | Comprehensive survey of LLM agent architectures | [arXiv](https://arxiv.org/abs/2308.11432) |
| **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** | 2023 | Jimenez et al. (Princeton) | Benchmark for real software engineering tasks | [arXiv](https://arxiv.org/abs/2310.06770) |

---

### Constitutional AI & RLHF

| Paper | Year | Authors | Key Contribution | Link |
|-------|------|---------|-----------------|------|
| **Constitutional AI: Harmlessness from AI Feedback** | 2022 | Bai et al. (Anthropic) | AI critiques itself against a constitution | [arXiv](https://arxiv.org/abs/2212.08073) |
| **Claude's Character** | 2023 | Anthropic | Values, safety, and helpfulness design for Claude | [Anthropic](https://www.anthropic.com/research/claude-character) |
| **Scaling Supervision: How Anthropic Trains Claude** | 2023 | Anthropic | CAI pipeline and training details | [Anthropic](https://www.anthropic.com/research) |
| **Reward Model Ensembles Help Mitigate Overoptimization** | 2023 | Eisenstein et al. (Google) | Ensemble-based reward modeling | [arXiv](https://arxiv.org/abs/2310.02743) |
| **RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback** | 2023 | Lee et al. (Google) | AI-generated feedback at scale | [arXiv](https://arxiv.org/abs/2309.00267) |
| **Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models** | 2024 | Chen et al. | SPIN: iterative self-play training | [arXiv](https://arxiv.org/abs/2401.01335) |

---

### Memory & Long-Horizon Reasoning

| Paper | Year | Authors | Key Contribution | Link |
|-------|------|---------|-----------------|------|
| **MemGPT: Towards LLMs as Operating Systems** | 2023 | Packer et al. (Berkeley) | Tiered memory management for long-context agents | [arXiv](https://arxiv.org/abs/2310.08560) |
| **Cognitive Architectures for Language Agents (CoALA)** | 2023 | Sumers et al. | Unifying framework: memory, action, decision-making | [arXiv](https://arxiv.org/abs/2309.02427) |
| **Long-Range Arena: A Benchmark for Efficient Transformers** | 2020 | Tay et al. (Google) | Evaluation of long-range dependencies | [arXiv](https://arxiv.org/abs/2011.04006) |
| **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** | 2023 | Gu & Dao | State space model alternative to attention | [arXiv](https://arxiv.org/abs/2312.00752) |
| **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | 2020 | Lewis et al. (Meta/UCL) | Combining retrieval with generation | [arXiv](https://arxiv.org/abs/2005.11401) |
| **HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs** | 2024 | Guti et al. | Graph-based episodic memory | [arXiv](https://arxiv.org/abs/2405.14831) |

---

### Formal Verification & Mathematical Reasoning

| Paper | Year | Authors | Key Contribution | Link |
|-------|------|---------|-----------------|------|
| **Solving Olympiad Geometry without Human Demonstrations (AlphaGeometry)** | 2024 | Trinh et al. (Google DeepMind) | LLM + symbolic deduction for olympiad geometry | [Nature](https://www.nature.com/articles/s41586-023-06747-5) |
| **Autoformalization with Large Language Models** | 2022 | Wu et al. (Google) | Translating math to formal proofs automatically | [arXiv](https://arxiv.org/abs/2205.12615) |
| **Minerva: Solving Quantitative Reasoning Problems with Language Models** | 2022 | Lewkowycz et al. (Google) | Math reasoning with step-by-step solutions | [arXiv](https://arxiv.org/abs/2206.14858) |
| **Formal Mathematics Statement Curriculum Learning (PACT)** | 2022 | Han et al. | Curriculum learning for theorem proving | [arXiv](https://arxiv.org/abs/2202.01344) |
| **Hypertree Proof Search for Neural Theorem Proving** | 2022 | Lample et al. (Meta) | MCTS-like proof search in Lean/Isabelle | [arXiv](https://arxiv.org/abs/2205.11491) |
| **AlphaProof & AlphaGeometry 2** | 2024 | Google DeepMind | Silver-medal level IMO performance | [DeepMind](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) |

---

## 🌱 Promising Research Areas

> These are the frontiers where significant breakthroughs are most likely in the next 3–5 years.

### 1. 🔍 Test-Time Compute Scaling
Allocating more computation at inference rather than just at training. Models like o1, o3, and DeepSeek-R1 demonstrate that "thinking longer" dramatically improves reasoning. **Key open question:** What's the optimal architecture for extended thinking — trees, chains, graphs, or something else?

**Why it matters:** Could be the next major paradigm shift after pretraining scaling. Compute at inference is fungible with training compute but allows dynamic allocation per problem difficulty.

**Key resources:**
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

---

### 2. 🧬 Mechanistic Interpretability at Scale
Understanding *why* models do what they do, not just *what* they output. Sparse autoencoders, activation patching, and circuit analysis are advancing rapidly. **Key open question:** Can we scale mech interp to frontier models? Can we catch deception?

**Why it matters:** The only path to verified alignment is interpretability. If we can "read" a model's computation, we can certify its safety properties.

**Key resources:**
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemanticity/index.html)
- [Scaling SAEs](https://arxiv.org/abs/2406.04093)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)

---

### 3. 🤝 Scalable Oversight & Weak-to-Strong Generalization
How do humans supervise AI systems smarter than themselves? Current RLHF is bottlenecked by human evaluator capability. **Key open question:** Can weaker models (or humans) effectively guide stronger models?

**Why it matters:** As AI becomes superintelligent, this becomes the central alignment problem.

**Key resources:**
- [Weak-to-Strong Generalization](https://arxiv.org/abs/2312.09390) (OpenAI, 2023)
- [Scalable Oversight via Debate](https://arxiv.org/abs/1805.00899)
- [IDA: Iterated Distillation and Amplification](https://arxiv.org/abs/1810.08575)

---

### 4. 🧩 Formal Reasoning & Verified AI
Combining the expressiveness of LLMs with the correctness guarantees of formal verification. **Key open question:** Can LLMs learn to produce machine-verifiable proofs reliably?

**Why it matters:** For high-stakes domains (medicine, law, software), we need AI outputs that can be *provably* correct, not just probably correct.

**Key resources:**
- [AlphaGeometry 2](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)
- [Lean4](https://github.com/leanprover/lean4)
- [Mathlib](https://github.com/leanprover-community/mathlib4)

---

### 5. 🌐 Multi-Agent Reasoning & Emergent Cooperation
Multiple specialized agents collaborating on tasks that exceed single-agent capability. **Key open question:** How do we ensure coherence, prevent error propagation, and align collective agent behavior?

**Why it matters:** Many real-world superintelligent tasks (scientific discovery, large-scale planning) require distributed cognition.

**Key resources:**
- [AutoGen](https://github.com/microsoft/autogen)
- [MetaGPT](https://github.com/geekan/MetaGPT)
- [OpenAI Swarm](https://github.com/openai/swarm)

---

### 6. 🔁 Self-Improvement & Recursive Learning
AI systems that improve their own training data, reasoning strategies, or model weights. **Key open question:** Can self-improvement be made safe and steerable?

**Why it matters:** Self-improvement loops could be the engine of superintelligence — and also the primary existential risk.

**Key resources:**
- [STaR](https://arxiv.org/abs/2203.14465)
- [Quiet-STaR](https://arxiv.org/abs/2403.09629)
- [SPIN](https://arxiv.org/abs/2401.01335)

---

### 7. 🔒 Deceptive Alignment & Robustness
Training a model to be aligned under normal conditions is insufficient if it can behave differently in deployment. **Key open question:** Can we detect and eliminate deceptive policies?

**Why it matters:** Deceptive alignment is one of Anthropic's core threat models for catastrophic AI risk.

**Key resources:**
- [Sleeper Agents](https://arxiv.org/abs/2401.05566)
- [Risks from Learned Optimization](https://arxiv.org/abs/1906.01820)
- [Eliciting Latent Knowledge](https://docs.google.com/document/d/1WwsnJQstPq91_Yh-Ch2XRL8H_EpsnjrC1dwZXR37PC0/)

---

### 8. 🧠 World Models & Causal Reasoning
LLMs struggle with genuine causal reasoning and world modeling. **Key open question:** Can we build models that maintain causal graphs and simulate counterfactuals reliably?

**Why it matters:** Human-level planning requires understanding cause and effect, not just statistical associations.

**Key resources:**
- [I-JEPA](https://arxiv.org/abs/2301.08243) (LeCun, Meta)
- [Causal Reasoning via LLMs Survey](https://arxiv.org/abs/2305.00050)
- [World Models](https://worldmodels.github.io/) (Ha & Schmidhuber)

---

### 9. 📐 Neurosymbolic AI & Hybrid Architectures
Combining neural networks with symbolic reasoning for robustness, interpretability, and generalization. **Key open question:** What's the right interface between neural and symbolic components?

**Why it matters:** Pure neural systems struggle with logical consistency; pure symbolic systems struggle with real-world uncertainty.

**Key resources:**
- [Neural Theorem Proving Survey](https://arxiv.org/abs/2212.10535)
- [DreamCoder](https://arxiv.org/abs/2006.08381)
- [AlphaCode](https://www.science.org/doi/10.1126/science.abq1158)

---

### 10. 🎯 Specification Gaming & Reward Hacking
Models optimize for the specified reward in ways that violate intended behavior. **Key open question:** How do we specify reward functions that are robust to optimization?

**Why it matters:** Goodhart's Law at scale — any measure becomes a target and ceases to be a good measure.

**Key resources:**
- [Specification Gaming Examples](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLui6T1XP8YpkUTls_OAzf1B98AKirMmJb5ChlC3xaKVIBaEGk8/pubhtml)
- [Avoiding Side Effects](https://arxiv.org/abs/1902.09725)
- [Conservative Agency](https://arxiv.org/abs/1902.09725)

---

## 💻 Notable GitHub Repositories

### Reasoning & Inference
| Repo | Description | Stars |
|------|-------------|-------|
| [microsoft/autogen](https://github.com/microsoft/autogen) | Multi-agent conversation framework | ⭐ 30k+ |
| [langchain-ai/langchain](https://github.com/langchain-ai/langchain) | Framework for LLM applications | ⭐ 90k+ |
| [hwchase17/langchainjs](https://github.com/langchain-ai/langchainjs) | JS port of LangChain | ⭐ 12k+ |
| [princeton-nlp/tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm) | Official Tree of Thoughts implementation | ⭐ 4k+ |
| [openai/evals](https://github.com/openai/evals) | Framework for evaluating LLMs | ⭐ 14k+ |
| [BerriAI/litellm](https://github.com/BerriAI/litellm) | Universal LLM API interface | ⭐ 12k+ |
| [openai/swarm](https://github.com/openai/swarm) | Lightweight multi-agent orchestration | ⭐ 16k+ |
| [geekan/MetaGPT](https://github.com/geekan/MetaGPT) | Multi-agent meta programming framework | ⭐ 43k+ |

### Interpretability & Analysis
| Repo | Description | Stars |
|------|-------------|-------|
| [neelnanda-io/TransformerLens](https://github.com/neelnanda-io/TransformerLens) | Mechanistic interpretability toolkit | ⭐ 5k+ |
| [EleutherAI/elk](https://github.com/EleutherAI/elk) | Eliciting latent knowledge experiments | ⭐ 1k+ |
| [jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) | Sparse autoencoder training library | ⭐ 600+ |
| [google-deepmind/penzai](https://github.com/google-deepmind/penzai) | Neural net visualization toolkit | ⭐ 1k+ |
| [baulab/rome](https://github.com/kmeng01/rome) | ROME: Rank-One Model Editing | ⭐ 2k+ |

### Formal Reasoning & Math
| Repo | Description | Stars |
|------|-------------|-------|
| [leanprover/lean4](https://github.com/leanprover/lean4) | Lean 4 theorem prover | ⭐ 4k+ |
| [leanprover-community/mathlib4](https://github.com/leanprover-community/mathlib4) | Lean 4 math library | ⭐ 1k+ |
| [openai/miniF2F](https://github.com/openai/miniF2F) | Math olympiad formal benchmark | ⭐ 500+ |
| [deepmind/alphageometry](https://github.com/google-deepmind/alphageometry) | AlphaGeometry: geometry theorem proving | ⭐ 3k+ |

### Training & Alignment
| Repo | Description | Stars |
|------|-------------|-------|
| [huggingface/trl](https://github.com/huggingface/trl) | Transformer Reinforcement Learning (PPO, DPO) | ⭐ 9k+ |
| [CarperAI/trlx](https://github.com/CarperAI/trlx) | Distributed RLHF training | ⭐ 4k+ |
| [anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook) | Anthropic API examples | ⭐ 6k+ |
| [openai/tiktoken](https://github.com/openai/tiktoken) | Fast tokenization | ⭐ 11k+ |
| [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Unified LLM eval framework | ⭐ 7k+ |

### Agentic Systems
| Repo | Description | Stars |
|------|-------------|-------|
| [Significant-Gravitas/AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) | Autonomous GPT-4 agent | ⭐ 165k+ |
| [joaomdmoura/crewAI](https://github.com/joaomdmoura/crewAI) | Role-based agent orchestration | ⭐ 20k+ |
| [cpacker/MemGPT](https://github.com/cpacker/MemGPT) | LLM with tiered memory | ⭐ 11k+ |
| [stanford-oval/storm](https://github.com/stanford-oval/storm) | Research paper generation agents | ⭐ 12k+ |

---

## 🧪 Benchmarks & Evaluations

| Benchmark | Domain | What It Tests | Link |
|-----------|--------|---------------|------|
| **MMLU** | General | Massive multitask language understanding (57 subjects) | [arXiv](https://arxiv.org/abs/2009.03300) |
| **GSM8K** | Math | Grade school math word problems | [arXiv](https://arxiv.org/abs/2110.14168) |
| **MATH** | Math | Competition math (AMC, AIME, Olympiad) | [arXiv](https://arxiv.org/abs/2103.03874) |
| **HumanEval** | Code | Python function synthesis from docstrings | [arXiv](https://arxiv.org/abs/2107.03374) |
| **SWE-bench** | Code | Real GitHub issue resolution | [arXiv](https://arxiv.org/abs/2310.06770) |
| **ARC-AGI** | Reasoning | Abstract reasoning / pattern recognition | [GitHub](https://github.com/fchollet/ARC-AGI) |
| **GPQA** | Science | PhD-level science questions | [arXiv](https://arxiv.org/abs/2311.12022) |
| **BIG-Bench** | Diverse | 204 tasks beyond model capabilities | [arXiv](https://arxiv.org/abs/2206.04615) |
| **HellaSwag** | Commonsense | Sentence completion | [arXiv](https://arxiv.org/abs/1905.07830) |
| **TruthfulQA** | Truthfulness | Measuring LLM hallucinations | [arXiv](https://arxiv.org/abs/2109.07958) |
| **miniF2F** | Math Proofs | Formal math olympiad problems | [GitHub](https://github.com/openai/miniF2F) |
| **APPS** | Code | Algorithm problem solving | [arXiv](https://arxiv.org/abs/2105.09938) |
| **AgentBench** | Agents | Multi-environment agent evaluation | [arXiv](https://arxiv.org/abs/2308.03688) |
| **HELMET** | Long-Context | Long context evaluation | [arXiv](https://arxiv.org/abs/2410.02669) |

---

## 🎓 Reading Lists by Level

### 🟢 Beginner (Start Here)
1. [A Beginner's Guide to AI/ML](https://ml-cheatsheet.readthedocs.io/en/latest/) — Core ML concepts
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual transformer explainer
3. [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) — Build GPT from scratch
4. [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) — First landmark reasoning paper
5. [Constitutional AI](https://arxiv.org/abs/2212.08073) — Alignment by self-critique

### 🟡 Intermediate
1. [Scaling Laws for NLP](https://arxiv.org/abs/2001.08361) — Understand training compute
2. [RLHF / InstructGPT](https://arxiv.org/abs/2203.02155) — Core alignment technique
3. [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) — Mechanistic basis for ICL
4. [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) — Feature compression
5. [Reflexion](https://arxiv.org/abs/2303.11366) — Agents that learn from mistakes
6. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — Process reward models
7. [DPO](https://arxiv.org/abs/2305.18290) — Alignment without RL

### 🔴 Advanced / Research Frontier
1. [Weak-to-Strong Generalization](https://arxiv.org/abs/2312.09390) — Core scalable oversight paper
2. [Sleeper Agents](https://arxiv.org/abs/2401.05566) — Deceptive alignment empirically
3. [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemanticity/index.html) — SAE-based interpretability
4. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — RL-trained reasoning models
5. [Scaling Test-Time Compute](https://arxiv.org/abs/2408.03314) — Inference-time scaling
6. [Mesa-optimization](https://arxiv.org/abs/1906.01820) — Inner vs. outer alignment
7. [AlphaGeometry 2](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) — Formal reasoning frontier
8. [Representation Engineering](https://arxiv.org/abs/2310.01405) — Behavior control via representations

### 📚 Essential Books
| Book | Author | Why Read It |
|------|--------|-------------|
| Superintelligence | Nick Bostrom | Foundational risks and pathways |
| The Alignment Problem | Brian Christian | History and current state |
| Human Compatible | Stuart Russell | CIRL and value alignment |
| Thinking, Fast and Slow | Daniel Kahneman | Human reasoning as baseline |
| Gödel, Escher, Bach | Douglas Hofstadter | Consciousness and self-reference |
| The Coming Wave | Mustafa Suleyman | Near-term AI trajectory |

---

## 🔭 Research Groups & Labs

### Safety-Focused
| Lab | Focus | Key Resource |
|-----|-------|-------------|
| [Anthropic](https://www.anthropic.com/research) | Constitutional AI, Interpretability, Alignment | [Research page](https://www.anthropic.com/research) |
| [ARC (Alignment Research Center)](https://www.alignment.org/) | ELK, Scalable Oversight | [Website](https://www.alignment.org/) |
| [MIRI (Machine Intelligence Research Institute)](https://intelligence.org/) | Decision Theory, Agent Foundations | [Papers](https://intelligence.org/research/) |
| [Center for AI Safety (CAIS)](https://www.safe.ai/) | X-risk, Policy, Research | [Website](https://www.safe.ai/) |
| [Redwood Research](https://www.redwoodresearch.org/) | Adversarial training, RLHF | [Papers](https://www.redwoodresearch.org/research) |

### Capability-Focused (with Safety Programs)
| Lab | Focus | Key Resource |
|-----|-------|-------------|
| [OpenAI](https://openai.com/research) | o1/o3 reasoning, GPT series | [Safety team](https://openai.com/safety) |
| [DeepMind](https://www.deepmind.com/research) | AlphaFold, Gemini, AlphaGeometry | [Research](https://www.deepmind.com/research) |
| [Meta AI (FAIR)](https://ai.meta.com/research/) | LLaMA, open research | [Papers](https://ai.meta.com/research/publications/) |
| [Google Brain / Google Research](https://research.google/) | PaLM, Gemini, T5 | [Blog](https://ai.googleblog.com/) |

### Academic
| Lab | Institution | Focus |
|-----|-------------|-------|
| [CHAI](https://humancompatible.ai/) | UC Berkeley | Value Alignment, IRL |
| [CS Department](https://nlp.stanford.edu/) | Stanford NLP | LLM Research |
| [Center for Human-Compatible AI](https://humancompatible.ai/) | Berkeley | Human-AI alignment |
| [Alignment Forum](https://www.alignmentforum.org/) | Community | AI Safety Discussion |
| [LessWrong](https://www.lesswrong.com/) | Community | Rationality + AI Risk |

---

## 💡 Open Problems & Ideas

These are high-impact open research questions where contributions could meaningfully advance the field:

### Reasoning
1. **Can we make process-level verification learnable without human annotation?** PRMs require expensive human labeling of reasoning steps. Can we generate synthetic step labels?
2. **What is the optimal "thinking budget" per problem type?** Current test-time scaling applies uniform compute; adaptive allocation could dramatically improve efficiency.
3. **How do we prevent reasoning collapse during RL training?** DeepSeek-R1 shows reasoning can emerge but also degrade — what regularizes this?

### Alignment
4. **Can sparse autoencoders detect deceptive reasoning in practice?** Theory suggests yes; empirical validation on intentionally deceptive models is open.
5. **What training procedures minimize the gap between inner and outer alignment?** Current methods don't address mesa-optimization directly.
6. **How do we specify preferences for novel situations the reward model has never seen?**

### Superintelligence
7. **What is the minimal cognitive architecture for recursive self-improvement?** Theoretical frameworks for safe, steerable self-improvement loops.
8. **Can formal verification be applied to large neural networks?** Neural network verification (e.g., via SMT solvers) currently scales only to tiny networks.
9. **What are the "intelligence genes" — the minimal set of capabilities required for human-level reasoning?**

### Multi-Agent
10. **How do we prevent collusion and reward hacking in multi-agent debates?** Debate assumes agents don't coordinate — but they might.
11. **What communication protocols maximize collective intelligence?** Human teams use meetings, documents, roles — what's the AI equivalent?

---

## 📰 Best Blogs & Newsletters

| Resource | Type | Focus |
|----------|------|-------|
| [Anthropic Research Blog](https://www.anthropic.com/research) | Blog | Safety, interpretability, Claude |
| [Alignment Forum](https://www.alignmentforum.org/) | Forum/Blog | AI safety research |
| [Distill.pub](https://distill.pub/) | Journal | ML interpretability (now archived but gold) |
| [Transformer Circuits Thread](https://transformer-circuits.pub/) | Blog | Mechanistic interpretability |
| [LessWrong](https://www.lesswrong.com/) | Forum | AI risk and rationality |
| [The Batch (deeplearning.ai)](https://www.deeplearning.ai/the-batch/) | Newsletter | Weekly AI news |
| [Import AI](https://jack-clark.net/) | Newsletter | Jack Clark's weekly AI update |
| [AI Alignment Newsletter](https://rohinshah.com/alignment-newsletter/) | Newsletter | Rohin Shah's curated summaries |
| [Sequoia's AI Ascent](https://www.sequoiacap.com/article-topic/ai/) | Blog | AI market and technology |
| [Andrej Karpathy's Blog](http://karpathy.github.io/) | Blog | Deep technical AI content |
| [Colah's Blog](http://colah.github.io/) | Blog | Neural network visualization |
| [Sebastian Raschka](https://sebastianraschka.com/blog/) | Blog | Practical LLM research |
| [interconnects.ai](https://www.interconnects.ai/) | Newsletter | RLHF and frontier models |

---

## 🤝 Contributing

Contributions are very welcome! This repo is designed to be a **community resource**, not a static list.

### How to Contribute
1. Fork this repository
2. Add your contribution (paper, repo, resource, open problem)
3. Ensure it follows the format of existing entries
4. Submit a Pull Request with a brief description

### Contribution Guidelines
- Papers should have clear relevance to reasoning or superintelligence research
- Include the year, authors, and a ≤2-sentence description of the key contribution
- Prefer arXiv, official lab blogs, or peer-reviewed sources
- For open problems, be specific — vague problems are less useful
- Star repos should have >100 stars unless they're new and exceptional

### What We're Looking For
- 📄 Recent papers (especially 2024–2025) we've missed
- 🔧 Useful implementation repos
- 🧩 New promising research areas
- 💡 Well-defined open problems
- 🌐 Non-English language resources and international research groups

---

## 📊 Repo Stats & Activity

This repository is actively maintained. To stay updated:
- ⭐ **Star** this repo to get notified of major updates
- 👁️ **Watch** for all changes
- 🔔 Subscribe to **Releases** for curated update summaries

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE). All linked papers and resources retain their original licenses.

---

## 🙏 Acknowledgments

Inspired by the incredible work from Anthropic, OpenAI, DeepMind, EleutherAI, and the broader AI safety and reasoning research community. Special thanks to the researchers who share their work openly.

---

> *"The question of whether a computer can think is no more interesting than the question of whether a submarine can swim."* — Edsger Dijkstra

> *"Intelligence is whatever machines haven't done yet."* — Larry Tesler

> *"The development of full artificial intelligence could spell the end of the human race... or it could be the best thing that ever happened to us."* — Stephen Hawking

---

<div align="center">
  <b>Made with ❤️ for the AI research community by Olaitan Olaleye</b><br>
  <i>If this helped you, please star ⭐ the repo and share it!</i>
</div>
