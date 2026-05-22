# NLPDL_group_1: Keeping LLMs on Track

This repository contains the code and annotation interface for the **Keeping LLMs on Track** project. 

The primary goal of this project is to investigate and annotate human-created "distractors" in task-oriented dialogues. We aim to understand how user utterances can intentionally or unintentionally steer Large Language Models (LLMs) away from their system instructions, and ultimately improve LLM alignment in multi-turn interactions.

## 🛠️ Setup & Installation

To ensure everyone is working in the same environment and to avoid dependency conflicts, please use a virtual environment running **Python 3.11**.

### 1. Create a Virtual Environment
Navigate to the root of the project directory in your terminal and create the environment:

**Mac/Linux:**
```bash
python3.11 -m venv .venv

```

**Windows:**

```bash
py -3.11 -m venv .venv

```

### 2. Activate the Environment

You must activate the environment every time you work on the project.

**Mac/Linux:**

```bash
source .venv/bin/activate

```

**Windows:**

```bash
.venv\Scripts\activate

```

### 3. Install Dependencies

Once the environment is active, install the required packages:

```bash
pip install -r requirements.txt

```

---

## 🚀 Running the Annotation Interface

We use a custom Streamlit GUI to make the data annotation process fast and efficient.

To launch the interface, ensure your virtual environment is active and run:

```bash
streamlit run interface.py

```

*(Note: A browser window should open automatically. If it doesn't, navigate to the `Local URL` provided in your terminal).*

## 📖 How to Use the Interface

1. **Load Data:** Use the left sidebar to load the base dataset directly from Hugging Face or upload an existing CSV of our annotations.
2. **Context:** The left panel displays the bot's system instructions and the base conversation flow.
3. **Annotate:** Use the right panel to craft multi-turn distractors designed to test the boundaries of the system instructions. You can also edit existing annotations to fix formatting or typos.
4. **Export:** Click the "Download Annotations (CSV)" button in the sidebar to save your work before closing the application!

```

```