# Enterprise Customer Support AI — ML + RAG Pipeline

> **PostgreSQL → Sentiment Model → RAG Retrieval → Groq Llama 3.1 → Streamlit UI**

An end-to-end intelligent customer support system for an e-commerce platform. It combines a locally trained Machine Learning sentiment model, a Retrieval-Augmented Generation (RAG) engine, and a Large Language Model into a single pipeline hosted on a Streamlit web application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Why This Architecture](#why-this-architecture)
- [Project Structure](#project-structure)
- [File-by-File Breakdown](#file-by-file-breakdown)
  - [database.py](#1-databasepy--database-layer)
  - [data_generator.py](#2-data_generatorpy--synthetic-data-generation)
  - [train_model.py](#3-train_modelpy--sentiment-analysis-model)
  - [rag_engine.py](#4-rag_enginepy--retrieval-augmented-generation)
  - [app.py](#5-apppy--streamlit-application)
  - [check_db.py](#6-check_dbpy--database-health-check)
- [End-to-End Data Flow](#end-to-end-data-flow)
- [Key Design Decisions](#key-design-decisions)
- [Setup & Execution Guide](#setup--execution-guide)

---

## Project Overview

This project takes historical e-commerce support tickets stored in a PostgreSQL database and does three things automatically:

1. **Analyses the emotional tone** of the customer's message (POSITIVE or NEGATIVE sentiment) using a locally trained ML model
2. **Retrieves the most relevant company policy** document based on the ticket content using semantic search
3. **Generates a professional, policy-grounded reply** that matches the customer's emotional state using Groq's Llama 3.1

The result is a support reply that is both tonally appropriate and factually accurate — grounded in real company policy rather than hallucinated information.

---

## Why This Architecture

A simple LLM chatbot has two core problems:

- It has no knowledge of your specific company policies, so it generates responses from general knowledge that may be inaccurate
- It responds the same way to an angry customer and a happy customer, which is poor customer service

This architecture solves both:

- The **sentiment model** ensures the tone of the reply is appropriate for the customer's mood
- The **RAG engine** ensures the content of the reply is grounded in real, accurate company policy

---

## Project Structure

```
GenAI/
│
├── database.py          # PostgreSQL connections, schema, data fetching
├── data_generator.py    # Generates 1,500 diverse synthetic support tickets
├── train_model.py       # Trains TF-IDF + Logistic Regression sentiment model
├── rag_engine.py        # Semantic policy retrieval using sentence-transformers
├── app.py               # Streamlit web application — ties everything together
├── check_db.py          # Quick database connection health check
├── requirements.txt     # All Python dependencies
└── .env                 # Credentials (not committed to Git)
```

---

## File-by-File Breakdown

### 1. `database.py` — Database Layer

The foundation of the project. Handles all communication between the application and PostgreSQL. Every other file that needs data goes through this module.

**What it does:**
- Reads database credentials from `.env` so sensitive information is never hardcoded
- Provides `get_connection()` — a single function any module can call for a live DB connection
- Creates the `support_tickets` table using `IF NOT EXISTS`, making it safe to call multiple times
- Fetches all tickets as a Pandas DataFrame via `fetch_all_tickets()`

**Table Schema:**

| Column | Type | Description |
|---|---|---|
| `ticket_id` | SERIAL PK | Auto-incrementing unique identifier |
| `timestamp` | TIMESTAMP | When the ticket was created |
| `customer_name` | VARCHAR(100) | Full name of the customer |
| `category` | VARCHAR(50) | Type of issue: Refunds, Delivery, Damaged, Payment, Feedback |
| `issue_description` | TEXT | The customer's message — core input to the ML pipeline |
| `rating` | INT | Satisfaction rating from 1 (worst) to 5 (best) |
| `order_value` | NUMERIC | Monetary value of the order |
| `status` | VARCHAR(20) | Open, In Progress, or Closed |
| `resolution_eta` | VARCHAR(50) | Expected resolution timeframe |
| `device_type` | VARCHAR(30) | Channel: Mobile App, Desktop Web, Phone Call, etc. |

---

### 2. `data_generator.py` — Synthetic Data Generation

Generates the 1,500 training records the ML model learns from. The quality and diversity of training data directly determines how well the model performs.

**The Problem with Fixed/Repeated Data:**

If 1,500 rows all repeat the same 8 sentences, the ML model simply memorises them. It achieves a fake 100% accuracy on training data but fails completely on any real input. This is called **overfitting**.

**The Solution — Faker-Filled Templates:**

Each template is filled with random order IDs, product names, amounts, and dates from the `Faker` library. Every single one of the 1,500 rows is genuinely unique text. The model learns to generalise from patterns, not memorise specific sentences.

**The Five Ticket Categories:**

| Category | Sentiment | Description |
|---|---|---|
| Delivery Delay | NEGATIVE | Orders arriving late, tracking issues, missing updates |
| Damaged Product | NEGATIVE | Items arriving broken, defective, or poorly packaged |
| Payment Issue | NEGATIVE | Double charges, failed transactions, coupon errors |
| Refunds & Returns | NEGATIVE | Return requests, refund delays, wrong items received |
| Feedback & Praise | POSITIVE | Compliments, fast delivery praise, agent shoutouts |

**Realistic Label Assignment:**

Ratings are not random. Negative categories receive low ratings (1–2 predominantly, occasionally 3). Positive tickets receive high ratings (4–5 mostly). This mirrors real-world data where not every unhappy customer gives a 1-star rating.

---

### 3. `train_model.py` — Sentiment Analysis Model

The Machine Learning core of the project. Reads ticket data, trains a text classifier to detect sentiment, evaluates it rigorously, and saves the trained model to disk.

#### What Is Sentiment Analysis?

Reading a piece of text and determining the emotional tone. In our case: is this customer frustrated (NEGATIVE) or satisfied (POSITIVE)? The LLM response changes based on this — empathetic and apologetic for upset customers, warm and enthusiastic for happy ones.

#### The Modelling Pipeline

**Step 1 — Data Preparation**

Tickets are labelled using their rating:
- Rating 4–5 → `POSITIVE` (label = 1)
- Rating 1–2 → `NEGATIVE` (label = 0)
- Rating 3 → **Dropped** — neutral ratings give ambiguous signal that adds noise

**Step 2 — TF-IDF Vectorisation**

ML models cannot process raw text — they only understand numbers. TF-IDF (Term Frequency – Inverse Document Frequency) converts each ticket into a numeric vector.

- **Term Frequency (TF):** How often a word appears in this specific ticket
- **Inverse Document Frequency (IDF):** How rare that word is across ALL tickets — rare words carry more meaning

Key setting — `ngram_range=(1, 2)`:

The model considers both single words AND two-word phrases. `'never arrived'` carries far more negative signal than `'never'` or `'arrived'` in isolation. `'exceeded expectations'` is clearly positive in a way neither word alone conveys.

**Step 3 — Logistic Regression Classifier**

Why Logistic Regression:
- Fast to train and fast to predict — suitable for real-time use
- Interpretable — you can inspect which words most strongly influence predictions
- Outputs probability scores via `predict_proba()`, not just binary labels
- Performs excellently on text classification with TF-IDF

`class_weight='balanced'` automatically compensates if one sentiment class has more training examples than the other, preventing bias.

**Step 4 — Model Evaluation**

| Metric | What It Tells Us |
|---|---|
| Test Accuracy | Percentage of tickets correctly classified on unseen data |
| ROC-AUC Score | How well the model separates the two classes. 1.0 = perfect, 0.5 = random guessing |
| 5-Fold Cross Validation | Trains and tests 5 times on different data splits — far more reliable than a single split |
| Confusion Matrix | 2×2 grid showing correct and incorrect predictions for each class, saved as PNG |

**Step 5 — Saving Artefacts**

The trained model, vectoriser, and metrics are saved as `.pkl` files. The Streamlit app loads these from disk on startup — taking milliseconds rather than retraining from scratch.

---

### 4. `rag_engine.py` — Retrieval-Augmented Generation

The most architecturally interesting component. RAG gives the LLM access to specific, accurate knowledge — in this case, your company's actual policy documents.

#### The Problem RAG Solves

**Without RAG:**
```
Customer:  "Can I return my order after 30 days?"
LLM Reply: "Most stores allow returns within 60 to 90 days..." ← hallucinated, wrong
```

**With RAG:**
```
Customer:  "Can I return my order after 30 days?"
LLM Reply: "Per our policy, returns are accepted within 30 days..." ← accurate
```

RAG grounds the LLM's response in real, verified information. It is the difference between a chatbot that sounds confident and one that **is correct**.

#### How RAG Works — Plain English

Think of the RAG engine as a very smart librarian. When a customer sends a message, the librarian does not search for keywords. Instead, it understands the *meaning* of the question and finds whichever policy document is most conceptually relevant — even if the exact words do not match.

For example: `"I was charged twice for my order"` correctly retrieves the Payment Policy even though that policy may never use the words `"charged twice"`. It retrieves it because the **meaning** matches.

#### The Three-Step Process

**Step 1 — Embedding (Converting Text to Meaning)**

The engine uses `all-MiniLM-L6-v2` from the `sentence-transformers` library. This model converts any text into a list of 384 numbers called a **vector** or **embedding**.

Two sentences that mean similar things produce vectors that are mathematically close to each other — even if they share no words. `"I was overcharged"` and `"I was billed twice"` produce very similar vectors.

At startup, all policy documents are embedded and stored in memory. This happens once and the results are cached.

**Step 2 — Cosine Similarity (Finding the Best Match)**

When a customer ticket arrives, it is embedded using the same model. The engine computes **cosine similarity** between the ticket vector and each policy document vector.

```
Cosine Similarity = how closely two vectors point in the same direction

Score close to 1.0 → highly related in meaning
Score close to 0.0 → largely unrelated
```

The engine returns the top 2 highest-scoring documents.

**Step 3 — Context Injection**

The top 2 retrieved policy documents are formatted as a clean text block and injected into the LLM's system prompt. The LLM sees the policy content alongside the customer's message and generates a response that references actual policy rather than inventing one.

#### Why Local Inference (Not an API)?

The original implementation called the HuggingFace Inference API and had a silent fallback that returned **random vectors** if the API failed — making retrieval completely meaningless with no warning to the user.

The current implementation runs `sentence-transformers` entirely locally:
- **Reliable** — no external API dependency, no rate limits
- **Fast** — no HTTP round-trip, local `encode()` takes milliseconds
- **Correct** — no silent failure mode; errors surface immediately

---

### 5. `app.py` — Streamlit Application

The main entry point. Brings every component together into a single interactive web application.

#### Startup & Caching

Three expensive resources are loaded at startup: the ML model, vectoriser, and RAG engine. Each is wrapped in `@st.cache_resource`:

```python
@st.cache_resource
def load_rag_engine():
    return RAGEngine()  # Loads once, reused for every user interaction
```

Without caching, the ML model would reload from disk every time a user clicks a button. With caching, it loads once and stays in memory for the entire session.

#### The Full Pipeline

| Step | Component | What Happens |
|---|---|---|
| 1 | Ticket Selection | User picks a ticket from PostgreSQL, optionally filtering by category |
| 2 | Sentiment Analysis | `predict_sentiment()` → TF-IDF vectorise → Logistic Regression → POSITIVE/NEGATIVE + confidence % |
| 3 | RAG Retrieval | Ticket embedded → cosine similarity computed → top 2 policies returned with similarity scores |
| 4 | LLM Response | Sentiment + policy text injected into Groq API call → Llama 3.1 generates reply |
| 5 | Display | Sentiment badge, confidence bar, retrieved policies (expandable), and AI reply shown |

#### The Two Tabs

**Tab 1 — Ticket Analyser**
Pick any database ticket, filter by category, and run the full pipeline with one click. All intermediate results are shown step by step, making the pipeline transparent and explainable.

**Tab 2 — Live Chat**
Type any message directly. The same three-step pipeline runs in real time. Chat history is preserved within the session using Streamlit's `st.session_state`.

---

### 6. `check_db.py` — Database Health Check

A lightweight utility script. Connects to the database, counts total rows, and prints a sample ticket. Run this any time you want to quickly confirm the database connection is working before running the full application.

```bash
python check_db.py
# Database Connection: SUCCESS
# Total Rows Found: 1500
# Sample Ticket: ID=1, Name=Jane Smith, Category=Delivery Delay
```

---

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────┐
│  STEP 1 — PostgreSQL Database                       │
│  Ticket text fetched from support_tickets table     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  STEP 2 — Sentiment Model                           │
│  TF-IDF Vectoriser → Logistic Regression            │
│  Output: POSITIVE / NEGATIVE + confidence %         │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  STEP 3 — RAG Engine                                │
│  Sentence Transformer → 384-dim vector              │
│  Cosine similarity vs. policy documents             │
│  Output: Top 2 most relevant policies               │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  STEP 4 — Groq API (Llama 3.1)                      │
│  System prompt = Sentiment tone + Policy context    │
│  User prompt   = Customer's original message        │
│  Output: Professional support reply                 │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│  STEP 5 — Streamlit UI                              │
│  Sentiment badge + confidence bar                   │
│  Retrieved policies with similarity scores          │
│  AI-generated reply                                 │
└─────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| TF-IDF + Logistic Regression | Fast, interpretable, no GPU required, excellent for binary text classification |
| `ngram_range=(1, 2)` | Two-word phrases like "never arrived" carry far more signal than individual words |
| `class_weight='balanced'` | Prevents bias toward whichever sentiment class has more training examples |
| Drop `rating == 3` | Neutral ratings give ambiguous signal that reduces model accuracy |
| `predict_proba()` for confidence | A percentage score is more useful than a binary label — builds trust in predictions |
| 5-fold cross validation | A single train/test split can be lucky or unlucky — 5 folds give a stable, honest estimate |
| Local sentence-transformers | Eliminates the HuggingFace API dependency and its silent random-vector failure mode |
| Cosine similarity | Normalised and length-agnostic — fair comparisons across short and long documents |
| Top-2 policy retrieval | Gives the LLM more context, covering tickets that relate to multiple policies |
| Groq + Llama 3.1 | Sub-second inference on an open-source model — makes live chat feel genuinely real-time |

---

## Setup & Execution Guide

### Prerequisites

- Python 3.9+
- PostgreSQL running locally
- Free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

If you are on an older Mac (no AVX support), set these before running:

```bash
export USE_TORCH=1
export TF_CPP_MIN_LOG_LEVEL=3
```

Or add them permanently to your `~/.zshrc`.

### 2. Create PostgreSQL Database

```sql
CREATE DATABASE customer_support_db;
```

### 3. Create `.env` File

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=customer_support_db
DB_USER=postgres
DB_PASSWORD=your_password_here
GROQ_API_KEY=your_groq_key_here
```

### 4. Run in Order

```bash
# Step 1 — Verify database connection
python check_db.py

# Step 2 — Generate 1,500 diverse tickets and insert into PostgreSQL
python data_generator.py

# Step 3 — Train the sentiment model (saves .pkl files + confusion_matrix.png)
python train_model.py

# Step 4 — Launch the Streamlit app
streamlit run app.py
```

### 5. Expected Terminal Output on First Launch

```
Loading sentence-transformers model locally...
Model loaded.
Initialising RAG Knowledge Base — indexing policy documents...
RAG Knowledge Base successfully indexed!
```

The `all-MiniLM-L6-v2` model (~80 MB) downloads on the first run and caches locally. Subsequent startups are fast.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Database | PostgreSQL + psycopg2 |
| Data Generation | Python Faker |
| ML — Vectorisation | scikit-learn TF-IDF |
| ML — Classification | scikit-learn Logistic Regression |
| RAG — Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| LLM | Groq API — Llama 3.1 8B Instant |
| Web UI | Streamlit |
| Environment | python-dotenv |

---

## Project Improvements Over Baseline

| Original Issue | Fix Applied |
|---|---|
| 8 unique sentences across 1,500 rows → model memorises, fake 100% accuracy | 10+ Faker-filled templates per category → every row is genuinely unique |
| `COMPANY_KNOWLEDGE_BASE.get(category)` — dictionary keyword lookup, not RAG | `RAGEngine.retrieve()` — cosine similarity over sentence embeddings |
| HuggingFace API for embeddings — silent random vector fallback on failure | Local `sentence-transformers` inference — reliable, fast, no API dependency |
| No confidence score — only POSITIVE / NEGATIVE label | `predict_proba()` → confidence float 0.0–1.0 displayed as percentage |
| No model evaluation beyond accuracy | ROC-AUC, 5-fold CV, confusion matrix PNG added |
| `df` standalone dead code line in `fetch_all_tickets()` | Removed |
| No live chat interface | Tab 2 added: real-time sentiment + RAG + LLM reply |