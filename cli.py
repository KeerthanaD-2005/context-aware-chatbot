#!/usr/bin/env python3
"""
Interactive Terminal CLI for StudyBuddy Context-Aware Chatbot.
Supports multi-turn conversations, live context window inspection,
and strategy switching directly in your terminal.
"""

import os
import sys
import uuid
from typing import Optional

from dotenv import load_dotenv

# Load environment before importing app components
load_dotenv()

from app import (
    DEFAULT_CONTEXT_STRATEGY,
    MAX_HISTORY_MESSAGES,
    SYSTEM_PROMPT,
    ConversationContextManager,
    LLMError,
    dispatch_llm,
    get_active_provider,
)

# Terminal ANSI colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner(provider: str, model: str, strategy: str):
    print(f"{CYAN}{BOLD}================================================================={RESET}")
    print(f"{CYAN}{BOLD}              StudyBuddy — Context-Aware AI Chatbot              {RESET}")
    print(f"{CYAN}{BOLD}================================================================={RESET}")
    print(f" {BOLD}Provider:{RESET} {provider}  |  {BOLD}Model:{RESET} {model}  |  {BOLD}Strategy:{RESET} {strategy}")
    print(f" {DIM}Context Limit: {MAX_HISTORY_MESSAGES} messages retained in active window{RESET}")
    print(f" {DIM}Commands: /new (reset chat), /stats (view context), /strategy, /exit{RESET}")
    print(f"{CYAN}================================================================={RESET}\n")


def print_stats(stats: dict, total_turns: int):
    print(f"\n{YELLOW}{BOLD}--- Context Window Stats ---{RESET}")
    print(f" Strategy:               {stats.get('strategy')}")
    print(f" Active context msgs:    {stats.get('messages_in_context')} / {stats.get('context_limit')}")
    print(f" Total conversation turns:{total_turns}")
    print(f" Older msgs truncated:   {stats.get('truncated_messages', 0)}")
    print(f" Running summary active: {stats.get('has_summary', False)}")
    print(f"{YELLOW}----------------------------{RESET}\n")


def run_cli():
    provider, model = get_active_provider()
    if provider == "none":
        print(f"{RED}{BOLD}Notice:{RESET} No API key detected. Please configure OPENAI_API_KEY or GEMINI_API_KEY in your .env file.")
        print(f"Example in .env:")
        print(f"  OPENAI_API_KEY=your_key_here")
        print(f"  OPENAI_MODEL=gpt-4o-mini\n")

    strategy = DEFAULT_CONTEXT_STRATEGY
    context_mgr = ConversationContextManager(max_messages=MAX_HISTORY_MESSAGES)
    conversation_id = str(uuid.uuid4())
    turn_count = 0

    print_banner(provider, model, strategy)

    while True:
        try:
            prompt_label = f"{GREEN}{BOLD}You [{turn_count + 1}]:{RESET} "
            user_input = input(prompt_label).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{CYAN}Goodbye! Have a great day learning with StudyBuddy.{RESET}")
            sys.exit(0)

        if not user_input:
            continue

        # Command handling
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print(f"{CYAN}Goodbye!{RESET}")
            break

        if user_input.lower() == "/new":
            conversation_id = str(uuid.uuid4())
            context_mgr.reset_conversation(conversation_id)
            turn_count = 0
            print(f"\n{MAGENTA}✦ Started a new conversation. Previous context cleared.{RESET}\n")
            continue

        if user_input.lower() == "/stats":
            data = context_mgr.get_conversation_data(conversation_id)
            # Reconstruct recent stats
            _, stats = context_mgr.build_context(conversation_id, "dummy", strategy)
            print_stats(stats, turn_count)
            if data.get("running_summary"):
                print(f"{DIM}Current Running Summary:\n{data['running_summary']}{RESET}\n")
            continue

        if user_input.lower().startswith("/strategy"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].lower() in ("sliding_window", "summarization"):
                strategy = parts[1].lower()
                print(f"{MAGENTA}✦ Switched context strategy to: {strategy}{RESET}\n")
            else:
                print(f"{YELLOW}Usage: /strategy sliding_window  OR  /strategy summarization{RESET}\n")
            continue

        if user_input.lower() == "/history":
            data = context_mgr.get_conversation_data(conversation_id)
            messages = data.get("messages", [])
            print(f"\n{YELLOW}{BOLD}--- Conversation History ({len(messages)} messages) ---{RESET}")
            for idx, msg in enumerate(messages, 1):
                role_col = GREEN if msg["role"] == "user" else MAGENTA
                print(f"{role_col}[{msg['role'].capitalize()} {idx}]:{RESET} {msg['content']}")
            print(f"{YELLOW}-------------------------------------------------------{RESET}\n")
            continue

        if user_input.lower() == "/help":
            print(f"\n{CYAN}Available Commands:{RESET}")
            print(f"  /new                  - Start a brand new conversation (fresh context)")
            print(f"  /stats                - Show context window usage & metrics")
            print(f"  /strategy [name]      - Switch strategy: 'sliding_window' or 'summarization'")
            print(f"  /history              - Print all previous messages in this conversation")
            print(f"  /exit, /quit          - Exit the chat\n")
            continue

        # Build context
        context_messages, stats = context_mgr.build_context(
            conversation_id=conversation_id,
            new_user_message=user_input,
            strategy=strategy,
        )

        print(f"{DIM}StudyBuddy is thinking...{RESET}", end="\r")

        try:
            reply_text, prov, mod = dispatch_llm(context_messages)
            # Clear thinking indicator line
            sys.stdout.write("\033[K")
            print(f"{MAGENTA}{BOLD}StudyBuddy:{RESET} {reply_text}\n")

            context_mgr.record_turn(conversation_id, user_input, reply_text)
            turn_count += 1

            # Print subtle context info
            ctx_kept = stats.get("messages_in_context", 0)
            ctx_limit = stats.get("context_limit", MAX_HISTORY_MESSAGES)
            strat_name = stats.get("strategy")
            truncated = stats.get("truncated_messages", 0)
            trunc_info = f", {truncated} older msgs truncated" if truncated > 0 else ""
            print(f"{DIM}[Context: {ctx_kept}/{ctx_limit} msgs in window ({strat_name}{trunc_info})]{RESET}\n")

        except LLMError as err:
            sys.stdout.write("\033[K")
            print(f"{RED}{BOLD}Error ({err.status_code}):{RESET} {err.message}\n")
        except Exception as exc:
            sys.stdout.write("\033[K")
            print(f"{RED}{BOLD}Unexpected Error:{RESET} {str(exc)}\n")


if __name__ == "__main__":
    run_cli()
