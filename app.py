import json
import logging
import os
import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# Load environment variables from .env
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ==============================================================================
# 1. Configuration & Persona Definition
# ==============================================================================

SYSTEM_PROMPT = """You are StudyBuddy, an enthusiastic, patient, and methodical learning assistant and Socratic study coach.

Your core traits and rules:
1. Role: Help learners master complex concepts by breaking them down into digestible, clear explanations.
2. Socratic Style: When appropriate, encourage critical thinking by asking a thought-provoking follow-up question.
3. Clarity & Examples: Use relatable everyday analogies, code snippets, or bullet points to illustrate ideas.
4. Honesty & Boundaries: If you are unsure or if information isn't provided in the conversation context, say so openly. Never invent or hallucinate facts.
5. Context Continuity: Actively acknowledge and build upon topics, preferences, or details shared earlier in this conversation.
6. Tone: Warm, supportive, intellectually stimulating, and concise.
"""

# Provider keys & models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Context window settings
DEFAULT_CONTEXT_STRATEGY = os.getenv("CONTEXT_STRATEGY", "sliding_window").strip().lower()
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
SUMMARY_TRIGGER_THRESHOLD = int(os.getenv("SUMMARY_TRIGGER_THRESHOLD", "8"))

# Initialize OpenAI client if library is installed and key is present
openai_client = None
if OPENAI_API_KEY and OPENAI_API_KEY.strip() and not OPENAI_API_KEY.startswith("put_your_"):
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY.strip(), timeout=30.0, max_retries=1)
    except Exception as e:
        logger.warning("Could not initialize OpenAI client: %s", e)


def get_active_provider() -> Tuple[str, str]:
    """Returns the detected active provider name and model."""
    explicit_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit_provider == "gemini" and GEMINI_API_KEY:
        return "gemini", GEMINI_MODEL
    if explicit_provider == "openai" and openai_client:
        return "openai", OPENAI_MODEL

    if openai_client:
        return "openai", OPENAI_MODEL
    if GEMINI_API_KEY and GEMINI_API_KEY.strip() and not GEMINI_API_KEY.startswith("put_your_"):
        return "gemini", GEMINI_MODEL
    return "none", "none"


# ==============================================================================
# 2. Context Window Management
# ==============================================================================

class ConversationContextManager:
    """
    Maintains multi-turn conversation histories per conversation_id with
    thread-safe operations and context window strategies:
    - 'sliding_window': Keeps the most recent N messages.
    - 'summarization': Compresses older messages beyond the window into a
      running summary and prepends it to the recent messages.
    """

    def __init__(self, max_messages: int = 10, summary_threshold: int = 8):
        self.max_messages = max_messages
        self.summary_threshold = summary_threshold
        self._conversations: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def get_or_create(self, conversation_id: str) -> Dict[str, Any]:
        with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = {
                    "messages": [],
                    "running_summary": "",
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }
            return self._conversations[conversation_id]

    def build_context(
        self, conversation_id: str, new_user_message: str, strategy: Optional[str] = None
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Builds the context payload for the LLM based on the chosen strategy.
        Returns:
            context_messages: List of message dicts (system, user, assistant).
            stats: Context statistics dictionary.
        """
        strat = (strategy or DEFAULT_CONTEXT_STRATEGY).strip().lower()
        if strat not in ("sliding_window", "summarization"):
            strat = "sliding_window"

        with self._lock:
            conv = self._conversations.setdefault(conversation_id, {
                "messages": [],
                "running_summary": "",
                "created_at": time.time(),
                "updated_at": time.time(),
            })
            history = list(conv["messages"])
            running_summary = conv.get("running_summary", "")

        total_previous = len(history)

        if strat == "summarization" and total_previous >= self.summary_threshold:
            # Active window is the last (max_messages - 2) messages
            window_size = max(4, self.max_messages - 2)
            recent_messages = history[-window_size:] if total_previous > window_size else history
            older_messages = history[:-window_size] if total_previous > window_size else []

            # If there are older messages not yet summarized, create or update running summary
            if older_messages and not running_summary:
                running_summary = self._create_extractive_summary(older_messages)
                with self._lock:
                    if conversation_id in self._conversations:
                        self._conversations[conversation_id]["running_summary"] = running_summary

            context_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if running_summary:
                context_messages.append({
                    "role": "system",
                    "content": f"[Context Summary of prior conversation]:\n{running_summary}",
                })
            context_messages.extend(recent_messages)
            context_messages.append({"role": "user", "content": new_user_message})

            stats = {
                "strategy": "summarization",
                "total_conversation_messages": total_previous + 1,
                "messages_in_context": len(context_messages) - 1,  # exclude system prompt
                "context_limit": self.max_messages,
                "has_summary": bool(running_summary),
                "summary_length": len(running_summary) if running_summary else 0,
                "truncated_messages": max(0, total_previous - len(recent_messages)),
            }
            return context_messages, stats

        else:
            # Default sliding window truncation
            recent_messages = history[-self.max_messages:] if total_previous > self.max_messages else history
            context_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            context_messages.extend(recent_messages)
            context_messages.append({"role": "user", "content": new_user_message})

            truncated = max(0, total_previous - len(recent_messages))
            stats = {
                "strategy": "sliding_window",
                "total_conversation_messages": total_previous + 1,
                "messages_in_context": len(recent_messages) + 1,
                "context_limit": self.max_messages,
                "has_summary": False,
                "truncated_messages": truncated,
            }
            return context_messages, stats

    def record_turn(self, conversation_id: str, user_message: str, assistant_reply: str) -> None:
        """Saves a completed user-assistant turn to the conversation history."""
        with self._lock:
            conv = self._conversations.setdefault(conversation_id, {
                "messages": [],
                "running_summary": "",
                "created_at": time.time(),
                "updated_at": time.time(),
            })
            conv["messages"].extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_reply},
            ])
            conv["updated_at"] = time.time()

            # Bound maximum in-memory storage to prevent unbounded memory growth (e.g. 50 turns max per session)
            if len(conv["messages"]) > 100:
                conv["messages"] = conv["messages"][-100:]

    def reset_conversation(self, conversation_id: str) -> None:
        with self._lock:
            if conversation_id in self._conversations:
                self._conversations[conversation_id] = {
                    "messages": [],
                    "running_summary": "",
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }

    def get_conversation_data(self, conversation_id: str) -> Dict[str, Any]:
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if not conv:
                return {"messages": [], "running_summary": "", "total": 0}
            return {
                "messages": list(conv["messages"]),
                "running_summary": conv.get("running_summary", ""),
                "total": len(conv["messages"]),
            }

    @staticmethod
    def _create_extractive_summary(messages: List[Dict[str, str]]) -> str:
        """
        Extracts salient points from older turns to create a concise context summary.
        Can run offline without requiring additional LLM tokens.
        """
        highlights = []
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"].replace("\n", " ").strip()
            if len(content) > 120:
                content = content[:117] + "..."
            highlights.append(f"- {role}: {content}")
        return "\n".join(highlights)


context_manager = ConversationContextManager(
    max_messages=MAX_HISTORY_MESSAGES,
    summary_threshold=SUMMARY_TRIGGER_THRESHOLD,
)


# ==============================================================================
# 3. LLM API Dispatchers & Robust Error Handling
# ==============================================================================

class LLMError(Exception):
    """Custom exception containing user-friendly error message and HTTP status code."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def call_openai_chat(messages: List[Dict[str, str]], model: str) -> str:
    """Calls OpenAI chat completion API with comprehensive error handling."""
    if openai_client is None:
        raise LLMError(
            "OpenAI API key is not configured. Add a valid OPENAI_API_KEY to your .env file and restart the server.",
            status_code=503,
        )

    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            NotFoundError,
            RateLimitError,
        )

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=30.0,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            answer = "I couldn't generate a text response this time. Please rephrase or try again."
        return answer

    except AuthenticationError:
        raise LLMError(
            "Authentication failed: Your OpenAI API key appears to be invalid or expired. "
            "Please check OPENAI_API_KEY in your .env file.",
            status_code=401,
        )
    except RateLimitError:
        raise LLMError(
            "OpenAI rate limit or quota exceeded. Please check your plan and billing details at platform.openai.com.",
            status_code=429,
        )
    except APITimeoutError:
        raise LLMError(
            "The OpenAI API took too long to respond (> 30 seconds). Please try again.",
            status_code=504,
        )
    except APIConnectionError:
        raise LLMError(
            "Could not connect to the OpenAI API. Please check your internet connection and try again.",
            status_code=502,
        )
    except NotFoundError as exc:
        raise LLMError(
            f"Model '{model}' was not found. Please check OPENAI_MODEL in .env (e.g. gpt-4o-mini).",
            status_code=404,
        )
    except APIStatusError as exc:
        logger.warning("OpenAI API status error: %s - %s", exc.status_code, exc.message)
        raise LLMError(
            f"The OpenAI API returned an error ({exc.status_code}): {exc.message}",
            status_code=502,
        )
    except Exception as exc:
        logger.exception("Unexpected OpenAI API call error")
        raise LLMError(
            f"An unexpected error occurred while communicating with OpenAI: {str(exc)}",
            status_code=500,
        )


def call_gemini_chat(messages: List[Dict[str, str]], model: str) -> str:
    """Calls Google Gemini REST API with comprehensive error handling."""
    api_key = GEMINI_API_KEY
    if not api_key:
        raise LLMError(
            "Gemini API key is not configured. Add GEMINI_API_KEY or GOOGLE_API_KEY to your .env file.",
            status_code=503,
        )

    # Convert standard role messages to Gemini format
    system_instruction = None
    contents = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            # Gemini supports system_instruction
            if system_instruction is None:
                system_instruction = {"parts": [{"text": content}]}
            else:
                system_instruction["parts"].append({"text": content})
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})

    payload: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        payload["system_instruction"] = system_instruction

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key.strip()}"
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30.0)
    except requests.exceptions.Timeout:
        raise LLMError("The Gemini API took too long to respond (> 30s). Please try again.", status_code=504)
    except requests.exceptions.ConnectionError:
        raise LLMError("Could not connect to the Gemini API. Please check your internet connection.", status_code=502)
    except Exception as exc:
        logger.exception("Network failure reaching Gemini")
        raise LLMError(f"Network error reaching Gemini API: {str(exc)}", status_code=502)

    if resp.status_code == 400:
        try:
            err_data = resp.json().get("error", {})
            err_msg = err_data.get("message", "Invalid request parameters.")
        except Exception:
            err_msg = resp.text
        raise LLMError(f"Gemini API error (400 Bad Request): {err_msg}", status_code=400)

    if resp.status_code in (401, 403):
        raise LLMError(
            "Authentication failed: Your Gemini API key is invalid or unauthorized. "
            "Please check GEMINI_API_KEY in your .env file.",
            status_code=401,
        )

    if resp.status_code == 429:
        raise LLMError(
            "Gemini rate limit reached or quota exhausted. Please try again in a moment.",
            status_code=429,
        )

    if not resp.ok:
        logger.warning("Gemini error status %s: %s", resp.status_code, resp.text)
        raise LLMError(
            f"The Gemini API returned error status {resp.status_code}. Please try again later.",
            status_code=502,
        )

    try:
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"].strip()
        return "I received an empty response from Gemini. Please try asking again."
    except Exception as exc:
        logger.exception("Failed to parse Gemini response")
        raise LLMError("Failed to parse response from Gemini API.", status_code=500)


def dispatch_llm(messages: List[Dict[str, str]]) -> Tuple[str, str, str]:
    """
    Selects the active LLM provider and generates a response.
    Returns:
        (reply_text, provider_name, model_name)
    """
    provider, model = get_active_provider()

    if provider == "openai":
        reply = call_openai_chat(messages, model)
        return reply, "OpenAI", model
    elif provider == "gemini":
        reply = call_gemini_chat(messages, model)
        return reply, "Gemini", model
    else:
        raise LLMError(
            "No API key configured. Please add OPENAI_API_KEY (or GEMINI_API_KEY) to your .env file and restart the server.",
            status_code=503,
        )


# ==============================================================================
# 4. HTTP Endpoints
# ==============================================================================

@app.get("/")
def home():
    """Serves the main chat application interface."""
    return render_template("index.html")


@app.get("/api/health")
def health():
    """Returns application health and current LLM provider configuration."""
    provider, model = get_active_provider()
    return jsonify({
        "status": "healthy",
        "provider": provider,
        "model": model,
        "context_strategy": DEFAULT_CONTEXT_STRATEGY,
        "max_history_messages": MAX_HISTORY_MESSAGES,
        "has_openai_key": bool(openai_client),
        "has_gemini_key": bool(GEMINI_API_KEY),
    })


@app.post("/api/chat")
def chat():
    """
    Main chat endpoint:
    - Validates message input
    - Manages context window (sliding window or summarization)
    - Dispatches to LLM API
    - Stores multi-turn history on success
    - Returns reply and detailed context window metrics
    """
    data = request.get_json(silent=True) or {}
    raw_message = data.get("message")
    conversation_id = (data.get("conversation_id") or "").strip() or str(uuid.uuid4())
    requested_strategy = data.get("strategy")

    # Input validation
    if not isinstance(raw_message, str) or not raw_message.strip():
        return jsonify({"error": "Please enter a message."}), 400

    clean_message = raw_message.strip()
    if len(clean_message) > 8000:
        return jsonify({
            "error": "That message is too long (limit is 8,000 characters). Please condense your message."
        }), 400

    # Build context snapshot with context management strategy
    context_messages, context_stats = context_manager.build_context(
        conversation_id=conversation_id,
        new_user_message=clean_message,
        strategy=requested_strategy,
    )

    # Call LLM API with full error handling
    try:
        reply_text, provider_name, model_name = dispatch_llm(context_messages)
    except LLMError as exc:
        return jsonify({"error": exc.message, "status_code": exc.status_code}), exc.status_code
    except Exception as exc:
        logger.exception("Unhandled error during chat completion")
        return jsonify({
            "error": "An unexpected error occurred while generating the reply. Please try again."
        }), 500

    # Record turn only after a successful response
    context_manager.record_turn(
        conversation_id=conversation_id,
        user_message=clean_message,
        assistant_reply=reply_text,
    )

    return jsonify({
        "conversation_id": conversation_id,
        "reply": reply_text,
        "provider": provider_name,
        "model": model_name,
        "context_stats": context_stats,
    })


@app.post("/api/new-chat")
def new_chat():
    """Generates a fresh conversation ID with independent context."""
    new_id = str(uuid.uuid4())
    context_manager.get_or_create(new_id)
    return jsonify({
        "conversation_id": new_id,
        "message": "Fresh conversation initialized.",
    })


@app.get("/api/conversation/<conversation_id>")
def get_conversation(conversation_id: str):
    """Inspects history and context statistics for a specific conversation."""
    data = context_manager.get_conversation_data(conversation_id)
    return jsonify({
        "conversation_id": conversation_id,
        "total_messages": data["total"],
        "running_summary": data["running_summary"],
        "messages": data["messages"],
    })


@app.delete("/api/conversation/<conversation_id>")
def delete_conversation(conversation_id: str):
    """Resets memory for a specific conversation."""
    context_manager.reset_conversation(conversation_id)
    return jsonify({
        "conversation_id": conversation_id,
        "message": "Conversation history cleared.",
    })


if __name__ == "__main__":
    provider, model = get_active_provider()
    logger.info("Starting StudyBuddy Chatbot on http://127.0.0.1:5000")
    logger.info("Active Provider: %s | Model: %s | Strategy: %s", provider, model, DEFAULT_CONTEXT_STRATEGY)
    app.run(host="127.0.0.1", port=5000, debug=True)
