
# 🔬 Jaddangi AI Lab: The "Glass Box" 0.5B Engine

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Edge AI](https://img.shields.io/badge/Edge_AI-Optimized-00C7B7?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

**Jaddangi-Alfa v1.6.9** is a pure, zero-dependency PyTorch implementation of a 500-million parameter Small Language Model (SLM). Built from the ground up, this engine strips away heavy high-level wrappers (like the standard Hugging Face `transformers` library) to provide total architectural transparency and extreme hardware efficiency.

Designed and engineered by **Kurumalla Venkataramana** at the **Jaddangi IT & AI Consultancy**.

---

## 🧠 The "Glass Box" Philosophy

Most AI developers treat models as "Black Boxes"—feeding text into an API or a `.generate()` function without understanding the underlying matrix operations. 

Jaddangi-Alfa is built as a **"Glass Box."** Every single layer, attention head, Rotary Positional Embedding (RoPE), and activation function is explicitly written in raw PyTorch. This architecture enables:
*   **Mechanistic Interpretability:** Pause the engine mid-generation to inspect or alter hidden states layer-by-layer.
*   **Surgical Modifications:** Inject custom logic directly into the KV-Cache or Attention matrices.
*   **Frugal Computing:** Run advanced inference entirely on constrained Edge architectures (e.g., Intel i3 CPUs) without requiring external GPU dependencies.

---

## 🚀 Key Architectural Achievements

*   **Bit-Accurate Mathematical Perfection:** Successfully grafted pre-trained Qwen2-0.5B weights into this custom chassis. Achieved a proven maximum logit difference of **0.00000000** against the original architecture.
*   **32K Long-Context Mastery:** Engineered to support a 32,768-token context window using Scaled Dot-Product Attention (SDPA) capable of precise "Needle In A Haystack" retrieval across massive documents.
*   **Surgical LoRA Forge:** Includes a custom Parameter-Efficient Fine-Tuning (PEFT) loop built directly into the engine's physics. Allows for high-end Instruction Tuning while freezing 99.8% of the base parameters, enabling training on low-VRAM consumer GPUs (like a 16GB T4).
*   **Autonomous Agentic Interceptor:** Equipped with a native ReAct (Reason + Act) loop. The engine can autonomously halt text generation, execute live Python tools, and inject factual logic back into its context window.
    *   🌐 **Live Search** (DuckDuckGo integration)
    *   🧮 **Deterministic Math Engine** (SymPy integration)
    *   📂 **Private Vector Vault** (Local RAG via Sentence-Transformers)

---

## 📂 Repository Structure

*Note: Massive `.safetensors` weight files are intentionally ignored via `.gitignore` to keep this repository lightweight and code-focused.*

```text
jaddangi-ai-lab/
├── jaddangi_engine.py       # The core AI physics (Attention, MLP, RoPE, RMSNorm)
├── jaddangi_agent.py        # The ReAct tool-interceptor loop & Toolbox
├── jaddangi_forge.py        # Custom LoRA injection and Fine-Tuning pipelines
├── README.md                # Project documentation
└── .gitignore               # Security and weight exclusions
