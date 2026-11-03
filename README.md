# StudyBuddy — Context-Aware AI Chatbot ✦

> A production-grade multi-turn conversational AI chatbot with **dual LLM provider support (OpenAI & Google Gemini)**, **intelligent context window management (Sliding Window & Running Summarization)**, a **defined Socratic study coach persona**, **comprehensive error handling**, and both **Web UI** and **Terminal CLI** interfaces.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Backend-Flask-black.svg)](https://flask.palletsprojects.com/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Gemini-emerald.svg)](https://openai.com)
[![Tests](https://img.shields.io/badge/Tests-14%2F14%20Passing-brightgreen.svg)]()

---

## 📋 Features & Assignment Checklist

| Requirement | Implementation Detail | Status |
| :--- | :--- | :---: |
| **Backend Endpoint** | `POST /api/chat` with input validation, LLM dispatch, and response formatting | ✅ Complete |
| **Multi-Turn Context** | Thread-safe conversation state per `conversation_id`; references earlier turns | ✅ Complete |
| **Context Window Management** | **Sliding Window** (last $N$ messages) **+ Running Summarization** (compresses older turns) | ✅ Complete |
| **System Prompt / Persona** | **StudyBuddy**: Socratic learning coach with structured guidelines and analogies | ✅ Complete |
| **Error Handling** | Dedicated handling for 401 Auth, 429 Rate Limits, 504 Timeouts, 502 Connection errors | ✅ Complete |
| **Security & API Keys** | Read strictly from `.env`; verified excluded in `.gitignore` | ✅ Complete |
| **Interfaces** | Modern, responsive **Web UI** + Interactive **Terminal CLI (`cli.py`)** | ✅ Complete |
| **Test Suite** | 14 unit and integration tests covering multi-turn context, window limits, and errors | ✅ Complete |

---

## 🧠 Context Window Management Strategies

LLMs have finite context limits, and sending unbounded histories increases latency, token costs, and risk of context overflow. StudyBuddy implements two strategies:

### 1. Sliding Window Truncation (Default)
- Retains only the most recent $N$ messages (configurable via `MAX_HISTORY_MESSAGES`, default 10).
- Older messages roll off gracefully while preserving the active dialogue context.
- System prompt and personality remain pinned at the root of the context payload.

### 2. Running Summarization (Stretch Goal Implemented!)
- When conversation length exceeds `SUMMARY_TRIGGER_THRESHOLD`, older turns rolling out of the window are condensed into a structured context summary.
- The summary is prepended to the active message window:
  ```
  [System Persona Instructions]
        ↓
  [System: Context Summary of prior conversation]
        ↓
  [Recent User/Assistant Messages (Active Window)]
        ↓
  [New User Query]
  ```
- Allows the bot to recall essential facts from 20+ turns ago without blowing up token usage!
- Toggle live in the Web UI dropdown, via CLI `/strategy`, or in `.env` (`CONTEXT_STRATEGY=summarization`).

---

## 🎭 Defined Persona: StudyBuddy

Configured with a dedicated system prompt:
```text
You are StudyBuddy, an enthusiastic, patient, and methodical learning assistant and Socratic study coach.

Your core traits and rules:
1. Role: Help learners master complex concepts by breaking them down into digestible, clear explanations.
2. Socratic Style: When appropriate, encourage critical thinking by asking a thought-provoking follow-up question.
3. Clarity & Examples: Use relatable everyday analogies, code snippets, or bullet points to illustrate ideas.
4. Honesty & Boundaries: If you are unsure or if information isn't provided in the conversation context, say so openly.
5. Context Continuity: Actively acknowledge and build upon topics, preferences, or details shared earlier in this conversation.
6. Tone: Warm, supportive, intellectually stimulating, and concise.
```

---

## 🛠️ Project Architecture

```
context_chatbot_project/
├── app.py              # Flask server, context manager, LLM dispatchers, REST endpoints
├── cli.py              # Interactive terminal CLI with context inspector
├── test_chatbot.py     # 14 unit & integration tests (context, limits, errors)
├── requirements.txt    # Python dependencies (Flask, openai, python-dotenv, requests)
├── .env.example        # Template for API keys and configuration
├── .gitignore          # Ignores .env, .venv, bytecode, and cache files
├── static/
│   ├── app.js          # Frontend controller, markdown renderer, context stats updater
│   └── style.css       # Clean, modern, responsive styles with context inspector card
└── templates/
    └── index.html      # Single-page web application UI
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ installed.
- An API key from either:
  - **OpenAI:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  - **Google Gemini (Free tier):** [aistudio.google.com](https://aistudio.google.com/)

### 2. Clone & Setup Environment

```powershell
# Navigate to project folder
cd context_chatbot_project

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key in `.env`

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env    # Windows PowerShell
# cp .env.example .env         # Linux/macOS
```

Open `.env` and insert your active API key:

**For OpenAI:**
```env
OPENAI_API_KEY=sk-proj-your-active-openai-key-here
OPENAI_MODEL=gpt-4o-mini
```

**For Google Gemini (Free Tier):**
```env
GEMINI_API_KEY=AIzaSy-your-gemini-key-here
GEMINI_MODEL=gemini-1.5-flash
```

*(Never commit `.env` to Git. It is already included in `.gitignore`.)*

---

## 💻 Running the Application

### Option A: Web Application
Start the Flask development server:
```powershell
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

- Real-time Context Inspector displays active message count, limit, and strategy.
- Switch between **Sliding Window** and **Running Summarization** on the fly.
- Click **＋ New Conversation** to start a separate session with clean context.

### Option B: Terminal CLI (`cli.py`)
Run the interactive console chatbot directly in PowerShell or Bash:
```powershell
python cli.py
```

CLI Commands:
- `/new` — Reset conversation and start fresh
- `/stats` — View current context window usage, limits, and running summary
- `/strategy [sliding_window|summarization]` — Switch context strategy
- `/history` — Print full conversation history
- `/exit` — Quit the CLI

---

## 🧪 Verification: 5-Turn Context Test

To confirm that multi-turn context persists correctly across replies, test these 5 turns:

| Turn | User Message | Expected Behavior |
| :---: | :--- | :--- |
| **1** | `My project is called StudyBuddy and it helps students learn Python.` | Bot acknowledges the project name and purpose. |
| **2** | `I want to build it using Flask.` | Bot acknowledges the Flask choice and connects it to StudyBuddy. |
| **3** | `What is my project called?` | **Context Check:** Bot accurately recalls **"StudyBuddy"** from Turn 1. |
| **4** | `Which framework did I choose?` | **Context Check:** Bot accurately recalls **"Flask"** from Turn 2. |
| **5** | `Summarize what I told you about my project.` | **Context Check:** Bot synthesizes both **StudyBuddy**, **Python**, and **Flask**. |

*Then click **"＋ New Conversation"** and ask "What is my project called?" — the bot will politely confirm it doesn't know, verifying complete context isolation between sessions.*

---

## 🛡️ Robust Error Handling Matrix

| Scenario | HTTP Code | User-Facing Message |
| :--- | :---: | :--- |
| **Missing API Key** | `503` | `"No API key configured. Please add OPENAI_API_KEY or GEMINI_API_KEY to your .env file."` |
| **Invalid / Expired Key** | `401` | `"Authentication failed: Your API key appears to be invalid or expired. Check .env."` |
| **Rate Limit / Quota Exceeded**| `429` | `"Rate limit or quota exceeded. Please check your API account usage."` |
| **API Timeout (> 30s)** | `504` | `"The AI service took too long to respond (> 30 seconds). Please try again."` |
| **Network Failure** | `502` | `"Could not connect to the AI service. Please check your internet connection."` |
| **Empty Message** | `400` | `"Please enter a message."` |
| **Overlong Message (> 8k chars)** | `400` | `"That message is too long (limit is 8,000 characters). Please condense your message."` |

---

## 🚦 Automated Test Suite

Run the full automated test suite (14 passing tests):

```powershell
python -m unittest test_chatbot.py -v
```

Tests include:
- `test_single_turn_context`: System prompt insertion and first turn format.
- `test_multi_turn_history_persistence`: Memory across consecutive turns.
- `test_sliding_window_truncation`: Window limit enforcement and message roll-off.
- `test_running_summarization_strategy`: Automatic summary generation and prepending.
- `test_session_isolation`: Conversation A and B isolation.
- `test_reset_conversation`: History clearing.
- `test_chat_endpoint_successful_multi_turn`: 5-turn endpoint test.
- `test_chat_error_handling_auth_failure`: 401 handling.
- `test_chat_error_handling_rate_limit`: 429 handling.
- `test_chat_validation_empty_message`: 400 validation.
- `test_chat_validation_overlong_message`: 8k char validation.
- `test_health_endpoint`: Healthcheck response.

---


