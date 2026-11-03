"""
Unit & Integration Test Suite for StudyBuddy Context-Aware Chatbot.
Verifies multi-turn history persistence, context window management,
sliding window truncation, running summarization, session isolation,
error handling, and API endpoints.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from app import (
    MAX_HISTORY_MESSAGES,
    SYSTEM_PROMPT,
    ConversationContextManager,
    LLMError,
    app,
    context_manager,
)


class TestConversationContextManager(unittest.TestCase):
    """Tests the context window management logic and strategies."""

    def setUp(self):
        self.mgr = ConversationContextManager(max_messages=6, summary_threshold=6)
        self.conv_id = "test-session-123"

    def test_single_turn_context(self):
        """Verifies a single turn correctly builds context with system prompt."""
        context_msgs, stats = self.mgr.build_context(self.conv_id, "Hello, world!", strategy="sliding_window")

        # Must include system prompt and the user message
        self.assertEqual(len(context_msgs), 2)
        self.assertEqual(context_msgs[0]["role"], "system")
        self.assertEqual(context_msgs[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(context_msgs[1]["role"], "user")
        self.assertEqual(context_msgs[1]["content"], "Hello, world!")
        self.assertEqual(stats["strategy"], "sliding_window")
        self.assertEqual(stats["messages_in_context"], 1)

    def test_multi_turn_history_persistence(self):
        """Verifies messages persist across multiple turns within the same conversation."""
        # Turn 1
        self.mgr.record_turn(self.conv_id, "My name is Alice", "Nice to meet you, Alice!")
        # Turn 2
        self.mgr.record_turn(self.conv_id, "I am studying Python", "Python is a versatile language!")

        # Next prompt
        context_msgs, stats = self.mgr.build_context(self.conv_id, "What is my name and study topic?")

        # Expected: System prompt + 2 user + 2 assistant + 1 new user = 6 messages
        self.assertEqual(len(context_msgs), 6)
        self.assertEqual(context_msgs[1]["content"], "My name is Alice")
        self.assertEqual(context_msgs[2]["content"], "Nice to meet you, Alice!")
        self.assertEqual(context_msgs[3]["content"], "I am studying Python")
        self.assertEqual(context_msgs[4]["content"], "Python is a versatile language!")
        self.assertEqual(context_msgs[5]["content"], "What is my name and study topic?")
        self.assertEqual(stats["total_conversation_messages"], 5)
        self.assertEqual(stats["truncated_messages"], 0)

    def test_sliding_window_truncation(self):
        """
        Verifies that when conversation exceeds max_messages,
        the oldest messages are truncated and only the most recent N are sent.
        """
        # Add 5 complete turns (10 messages total)
        for i in range(1, 6):
            self.mgr.record_turn(self.conv_id, f"User message {i}", f"Assistant reply {i}")

        # Manager max_messages is 6
        context_msgs, stats = self.mgr.build_context(self.conv_id, "New turn", strategy="sliding_window")

        # Expect: 1 System prompt + 6 recent messages + 1 new user message = 8 messages
        self.assertEqual(len(context_msgs), 8)
        self.assertEqual(stats["strategy"], "sliding_window")
        self.assertEqual(stats["context_limit"], 6)
        self.assertEqual(stats["truncated_messages"], 4)  # 10 - 6 = 4 truncated

        # The oldest message in context must NOT be "User message 1"
        self.assertNotIn("User message 1", [m["content"] for m in context_msgs])
        # The most recent prior message must be present
        self.assertIn("Assistant reply 5", [m["content"] for m in context_msgs])

    def test_running_summarization_strategy(self):
        """
        Verifies that the summarization strategy creates a running summary of older messages
        and prepends it along with the active window.
        """
        # Add 5 complete turns (10 messages)
        for i in range(1, 6):
            self.mgr.record_turn(self.conv_id, f"Topic point {i}", f"Explanation point {i}")

        context_msgs, stats = self.mgr.build_context(self.conv_id, "Follow-up question", strategy="summarization")

        self.assertEqual(stats["strategy"], "summarization")
        self.assertTrue(stats["has_summary"])

        # Check that a system message with context summary is present
        roles = [m["role"] for m in context_msgs]
        self.assertIn("system", roles)
        contents = [m["content"] for m in context_msgs]
        has_summary_text = any("[Context Summary of prior conversation]" in c for c in contents)
        self.assertTrue(has_summary_text)

    def test_session_isolation(self):
        """Verifies that separate conversation IDs maintain completely isolated histories."""
        session_a = "user-alice-session"
        session_b = "user-bob-session"

        self.mgr.record_turn(session_a, "My favorite color is Blue", "I love Blue!")
        self.mgr.record_turn(session_b, "My favorite color is Red", "Red is vibrant!")

        data_a = self.mgr.get_conversation_data(session_a)
        data_b = self.mgr.get_conversation_data(session_b)

        self.assertEqual(len(data_a["messages"]), 2)
        self.assertEqual(len(data_b["messages"]), 2)
        self.assertIn("Blue", data_a["messages"][0]["content"])
        self.assertIn("Red", data_b["messages"][0]["content"])
        self.assertNotIn("Red", data_a["messages"][0]["content"])
        self.assertNotIn("Blue", data_b["messages"][0]["content"])

    def test_reset_conversation(self):
        """Verifies resetting a conversation clears its history."""
        self.mgr.record_turn(self.conv_id, "Test message", "Test reply")
        self.assertEqual(len(self.mgr.get_conversation_data(self.conv_id)["messages"]), 2)

        self.mgr.reset_conversation(self.conv_id)
        self.assertEqual(len(self.mgr.get_conversation_data(self.conv_id)["messages"]), 0)


class TestFlaskEndpoints(unittest.TestCase):
    """Tests the Flask API endpoints, input validation, and HTTP responses."""

    def setUp(self):
        self.client = app.test_client()
        self.app_ctx = app.app_context()
        self.app_ctx.push()

    def tearDown(self):
        self.app_ctx.pop()

    def test_home_page(self):
        """Verifies the GET / endpoint returns 200 with HTML UI."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"StudyBuddy", res.data)

    def test_health_endpoint(self):
        """Verifies the GET /api/health endpoint returns health status and config."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("provider", data)
        self.assertIn("model", data)
        self.assertIn("context_strategy", data)

    def test_new_chat_endpoint(self):
        """Verifies POST /api/new-chat generates a fresh UUID conversation_id."""
        res = self.client.post("/api/new-chat")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("conversation_id", data)
        self.assertTrue(len(data["conversation_id"]) > 10)

    def test_chat_validation_empty_message(self):
        """Verifies 400 Bad Request when message is empty or whitespace."""
        res = self.client.post("/api/chat", json={"message": "   "})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("Please enter a message", data["error"])

    def test_chat_validation_overlong_message(self):
        """Verifies 400 Bad Request when message exceeds 8,000 characters."""
        long_message = "x" * 8005
        res = self.client.post("/api/chat", json={"message": long_message})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("too long", data["error"])

    @patch("app.dispatch_llm")
    def test_chat_endpoint_successful_multi_turn(self, mock_dispatch):
        """
        Simulates a full 5-turn conversation via the POST /api/chat endpoint
        and verifies context persistence and returned statistics.
        """
        conversation_id = "test-endpoint-5turns"

        # Mock responses for 5 turns
        mock_replies = [
            "Great! StudyBuddy is an awesome project name.",
            "Flask is a fantastic, lightweight Python web framework.",
            "Your project is called StudyBuddy!",
            "You chose Flask for your web framework.",
            "To summarize: you are building StudyBuddy using Flask in Python.",
        ]

        turns_input = [
            "My project is called StudyBuddy and it helps students learn Python.",
            "I want to build it using Flask.",
            "What is my project called?",
            "Which framework did I choose?",
            "Summarize what I told you about my project.",
        ]

        for turn_idx, (user_msg, mock_reply) in enumerate(zip(turns_input, mock_replies), 1):
            mock_dispatch.return_value = (mock_reply, "MockProvider", "mock-model")

            res = self.client.post("/api/chat", json={
                "message": user_msg,
                "conversation_id": conversation_id,
            })

            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data["conversation_id"], conversation_id)
            self.assertEqual(data["reply"], mock_reply)
            self.assertIn("context_stats", data)
            expected_total = 2 * (turn_idx - 1) + 1
            self.assertEqual(data["context_stats"]["total_conversation_messages"], expected_total)

        # Inspect stored conversation endpoint
        inspect_res = self.client.get(f"/api/conversation/{conversation_id}")
        self.assertEqual(inspect_res.status_code, 200)
        inspect_data = inspect_res.get_json()
        # 5 turns = 10 messages (5 user + 5 assistant)
        self.assertEqual(inspect_data["total_messages"], 10)

    @patch("app.dispatch_llm")
    def test_chat_error_handling_auth_failure(self, mock_dispatch):
        """Verifies friendly 401 error message when API key authentication fails."""
        mock_dispatch.side_effect = LLMError(
            "Authentication failed: Your API key appears to be invalid or expired.",
            status_code=401,
        )

        res = self.client.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(res.status_code, 401)
        data = res.get_json()
        self.assertIn("Authentication failed", data["error"])

    @patch("app.dispatch_llm")
    def test_chat_error_handling_rate_limit(self, mock_dispatch):
        """Verifies friendly 429 error message when rate limit or quota is exceeded."""
        mock_dispatch.side_effect = LLMError(
            "Rate limit or quota exceeded. Please check your account usage.",
            status_code=429,
        )

        res = self.client.post("/api/chat", json={"message": "Hello"})
        self.assertEqual(res.status_code, 429)
        data = res.get_json()
        self.assertIn("Rate limit or quota exceeded", data["error"])


if __name__ == "__main__":
    unittest.main()
