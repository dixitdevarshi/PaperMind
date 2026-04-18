"""
answer_generator.py
Generates grounded answers using Claude API with source attribution,
conversation memory, and automatic language detection/response.
"""

import sys
from typing import Generator

import anthropic
from langchain.memory import ConversationBufferMemory

from src.logger import get_logger
from src.exception import LLMError
from src.components.retriever import RetrievedChunk

logger = get_logger(__name__)

SONNET_MODEL = "claude-sonnet-4-6"
HAIKU_MODEL  = "claude-haiku-4-5-20251001"
MAX_TOKENS   = 1024

SYSTEM_PROMPT = """You are PaperMind, an intelligent document assistant.

Your job is to answer questions based ONLY on the provided document context.

Rules:
1. Always respond in the SAME LANGUAGE the user used to ask the question.
2. Base your answer strictly on the provided context — do not use outside knowledge.
3. If the context does not contain enough information to answer, say so clearly.
4. Always cite your sources: mention the document name and page number.
5. Be concise and precise. Do not pad your answer with unnecessary text.
6. If comparing across documents, clearly label which document each point comes from.
7. If the user's question is a follow-up (e.g. 'in detail', 'explain more', 'elaborate'),
   answer using the document context and conversation history provided."""


class AnswerGenerator:
    """
    Generates answers using Claude API with retrieved context.

    Usage:
        generator = AnswerGenerator()
        response  = generator.generate(query, chunks)
    """

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        logger.info("AnswerGenerator ready")

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> dict:
        """
        Generate a grounded answer for the query using retrieved chunks.

        Returns dict with keys: answer, sources, confidence, model_used
        """
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

    def generate_stream(self, query: str, chunks: list[RetrievedChunk]) -> Generator[str, None, None]:
        """
        Streaming version of generate().
        Yields answer tokens one by one as they arrive from Claude.
        """
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

        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message("".join(full_answer))

    def clear_memory(self) -> None:
        self.memory.clear()
        logger.info("Conversation memory cleared")

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
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
        messages = self.memory.chat_memory.messages
        if not messages:
            return ""

        lines = ["CONVERSATION HISTORY:"]
        for msg in messages[-6:]:
            role = "User" if msg.type == "human" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _build_user_message(self, query: str, context_block: str, history: str) -> str:
        parts = []
        if history:
            parts.append(history)
            parts.append("")
        parts.append(context_block)
        parts.append("")
        parts.append(f"USER QUESTION: {query}")
        return "\n".join(parts)

    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[dict]:
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