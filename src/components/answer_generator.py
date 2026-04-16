import sys
from typing import Generator

import anthropic
from langchain.memory import ConversationBufferMemory

from src.logger import get_logger
from src.exception import LLMError
from src.components.retriever import RetrievedChunk

logger = get_logger(__name__)

# ── Model config ─────────────────────────────────────────────────────────────

SONNET_MODEL = "claude-sonnet-4-6"
HAIKU_MODEL  = "claude-haiku-4-5-20251001"
MAX_TOKENS   = 1024

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are PaperMind, an intelligent document assistant.

Your job is to answer questions based ONLY on the provided document context.

Rules:
1. Always respond in the SAME LANGUAGE the user used to ask the question.
2. Base your answer strictly on the provided context — do not use outside knowledge.
3. If the context does not contain enough information to answer, say so clearly.
4. Always cite your sources: mention the document name and page number.
5. Be concise and precise. Do not pad your answer with unnecessary text.
6. If comparing across documents, clearly label which document each point comes from."""

# ── Main class ───────────────────────────────────────────────────────────────

class AnswerGenerator:

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        logger.info("AnswerGenerator ready")

    # ── Public API ───────────────────────────────────────────

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> dict:

        if not query.strip():
            raise LLMError("Query cannot be empty", sys)

        context_block = self._build_context(chunks)
        history       = self._get_history()
        user_message  = self._build_user_message(query, context_block, history)

        logger.info(f"Generating answer for: '{query[:80]}'")

        try:
            message = self.client.messages.create(
                model=SONNET_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            answer = message.content[0].text
        except Exception as e:
            raise LLMError(f"Claude API call failed: {e}", sys)

        # Save to memory
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(answer)

        sources    = self._build_sources(chunks)
        confidence = chunks[0].confidence if chunks else "Low"

        logger.info(f"Answer generated — confidence: {confidence}, sources: {len(sources)}")

        return {
            "answer":     answer,
            "sources":    sources,
            "confidence": confidence,
            "model_used": SONNET_MODEL,
        }

    def generate_stream(
        self,
        query:  str,
        chunks: list[RetrievedChunk],
    ) -> Generator[str, None, None]:
        if not query.strip():
            raise LLMError("Query cannot be empty", sys)

        context_block = self._build_context(chunks)
        history       = self._get_history()
        user_message  = self._build_user_message(query, context_block, history)

        logger.info(f"Streaming answer for: '{query[:80]}'")

        full_answer = []

        try:
            with self.client.messages.stream(
                model=SONNET_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    full_answer.append(text)
                    yield text
        except Exception as e:
            raise LLMError(f"Claude streaming failed: {e}", sys)

        # Save complete answer to memory after streaming finishes
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message("".join(full_answer))

    def clear_memory(self) -> None:
        """Reset conversation memory — call when user starts a new session."""
        self.memory.clear()
        logger.info("Conversation memory cleared")

    # ── Internal ─────────────────────────────────────────────

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a numbered context block for the prompt."""
        if not chunks:
            return "No relevant context found in the loaded documents."

        lines = ["DOCUMENT CONTEXT:", ""]
        for i, chunk in enumerate(chunks, 1):
            lines.append(
                f"[{i}] Source: {chunk.source} | Page: {chunk.page} | "
                f"Section: {chunk.section} | Confidence: {chunk.confidence}"
            )
            lines.append(chunk.text)
            lines.append("")

        return "\n".join(lines)

    def _get_history(self) -> str:
        """Format conversation history for inclusion in the prompt."""
        messages = self.memory.chat_memory.messages
        if not messages:
            return ""

        lines = ["CONVERSATION HISTORY:"]
        for msg in messages[-6:]:   # last 3 turns (user + assistant × 3)
            role = "User" if msg.type == "human" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _build_user_message(
        self,
        query:         str,
        context_block: str,
        history:       str,
    ) -> str:
        """Assemble the full user message sent to Claude."""
        parts = []
        if history:
            parts.append(history)
            parts.append("")
        parts.append(context_block)
        parts.append("")
        parts.append(f"USER QUESTION: {query}")
        return "\n".join(parts)

    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
        """Build source attribution list from retrieved chunks."""
        seen    = set()
        sources = []

        for chunk in chunks:
            key = (chunk.source, chunk.page)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document":   chunk.source,
                    "page":       chunk.page,
                    "section":    chunk.section,
                    "confidence": chunk.confidence,
                    "score":      chunk.score,
                })

        return sources