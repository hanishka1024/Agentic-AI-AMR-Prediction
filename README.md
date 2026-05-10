# 🧬 Agentic AI-Based AMR Prediction Framework

> **An autonomous multi-agent system designed to predict Antimicrobial Resistance (AMR) by integrating genomic sequences and clinical data with medical-grade interpretability.**

---

### 🌟 Highlights

* 🤖 **Autonomous Multi-Agent Workflow:** Powered by **LangChain** to orchestrate four specialized agents for data, prediction, and narrative generation.
* 🧠 **State-of-the-Art LLM:** Utilizes **LLaMA 3.1 (70B)** via Groq for high-fidelity clinical interpretations.
* 📊 **Hybrid Ensemble Intelligence:** Combines **XGBoost, CatBoost, and LightGBM** for high-precision multi-class classification (Resistant, Intermediate, Susceptible).
* 💊 **Clinical-First Design:** Bridges the gap between raw genomic data and actionable clinician insights through natural language.
* 🚀 **Fast & Scalable:** Built with a modular architecture ready for real-time diagnostic environments.

---

### ℹ️ Overview

Antimicrobial resistance (AMR) is a top-tier global public health threat. Traditional diagnostic methods often require manual, time-consuming interpretation of complex genomic data. This project introduces an **Agentic AI Framework** that automates and interprets this process.

The system employs a "Chain-of-Thought" approach where an **Orchestrator Agent** coordinates specialized AI agents. This ensures that the final output is not just a statistical label, but a transparent and interpretable medical narrative.

#### 🏗️ The Architecture
The framework is divided into four distinct agents:
1.  **Agent 1 (The Engineer):** Handles data pre-processing and feature engineering of genomic signals.
2.  **Agent 2 (The Predictor):** Executes the ensemble machine learning models for classification.
3.  **Agent 3 (The Interpreter):** Uses LLaMA 3.1 to translate raw predictions into clinical narratives.
4.  **Agent 4 (The Orchestrator):** Governs the sequential logic and ensures seamless cross-agent communication.

---

### 🚀 Usage

The project features a **Streamlit** dashboard that allows users to upload data and view agentic reasoning in real-time.

```python
# The core logic is powered by a LangChain Orchestrator
from langchain_groq import ChatGroq

# Initialize the Clinical Interpreter Agent
llm = ChatGroq(model_name="llama-3.1-70b-versatile")
response = llm.invoke("Generate a clinician-curated explanation for the detected resistance...")
print(response.content)
```

---

## ⬇️ Installation

### Requirements
- Python 3.10+
- Groq API Key

---

### Clone the Repository

```bash
git clone https://github.com/hanishka1024/Agentic-AI-AMR-Prediction.git

cd Agentic-AI-AMR-Prediction
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Configure API Key

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_actual_key_here
```

---

### Run the Application

```bash
streamlit run app.py
```

---
## 💭 Feedback & Contributions

Contributions and suggestions are welcome.

You can:

- ⭐ Star the repository
- 🐛 Open an issue
- 💡 Suggest improvements
- 🔀 Submit a pull request

---

## 📜 License

This project is intended for academic and research purposes.

---

## ✨ Final Note

> “AI-powered clinical systems can help accelerate antimicrobial resistance detection and support faster, evidence-based treatment decisions.”
