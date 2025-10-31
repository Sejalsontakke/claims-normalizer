# 🏥 claims-normalizer

Project to convert raw, unstructured insurance claim notes into a standardized JSON format. It features a Hugging Face T5 fine-tuned model and an alternative LLM API prompting approach, wrapped in a high-performance FastAPI service.

## 🚀 Key Features

* **Dual ML Approach:** Supports fine-tuning a T5-base model (S2S) and using zero/few-shot prompting with an instruction-following LLM (e.g., GPT-4, Gemini, Claude).
* **Structured Output:** Normalizes messy text into a reliable JSON schema.
* **Production Ready:** Deployed via **FastAPI** and **Docker** for containerization.

## 📦 Repository Structure