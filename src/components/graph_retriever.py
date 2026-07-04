import re
import sys
from dataclasses import dataclass
from pathlib import Path

import spacy
import networkx as nx

from src.logger import get_logger
from src.exception import RetrievalError
from src.components.graph_builder import GraphBuilder, GRAPH_PATH, RELEVANT_ENTITY_TYPES, LEADING_ARTICLES

logger = get_logger(__name__)

DEFAULT_HOPS = 2


@dataclass
class GraphRetrievedChunk:
    """Single chunk retrieved via graph traversal, with the path that found it."""
    text:        str
    source:      str
    page:        str | int
    entity_path: list[str]   # the chain of entities that led to this chunk
    relation:    str         # the relationship label on the edge that found it


class GraphRetriever:
    """
    Retrieves document chunks by traversing the knowledge graph from
    entities found in the query, instead of using vector similarity.

    Usage:
        retriever = GraphRetriever()
        chunks    = retriever.retrieve("How does Volkswagen relate to Porsche?", hops=2)
    """

    def __init__(self, graph_path: str | None = None, model_name: str = "en_core_web_sm"):
        logger.info(f"Loading spaCy model for query entity extraction: {model_name}")
        self.nlp = spacy.load(model_name)

        load_path = Path(graph_path) if graph_path else GRAPH_PATH
        if not load_path.exists():
            raise RetrievalError(
                f"No knowledge graph found at: {load_path}. "
                f"Run graph_builder.py first to build one.", sys
            )

        builder    = GraphBuilder.__new__(GraphBuilder)  # skip __init__, we only need .load()
        self.graph = builder.load(str(load_path))

        # Pre-build a lowercase lookup so exact matching is fast and case-insensitive
        self._node_lookup: dict[str, str] = {
            node.lower(): node for node in self.graph.nodes()
        }

        logger.info(f"GraphRetriever ready — graph has {self.graph.number_of_nodes()} nodes")

    def retrieve(self, query: str, hops: int = DEFAULT_HOPS, n_results: int = 5) -> list[GraphRetrievedChunk]:
        """
        Retrieve chunks by:
          1. Extracting entities from the query using spaCy NER
          2. Normalizing them the same way the graph's nodes were normalized
          3. Finding exact (case-insensitive) matches in the graph
          4. Traversing N hops outward to find connected entities
          5. Returning the chunks associated with the edges traversed

        Parameters:
            query     : user question
            hops      : how many relationship hops to traverse (default 2)
            n_results : max number of chunks to return

        Returns:
            list[GraphRetrievedChunk] — empty list if no query entities
            are found in the graph at all
        """
        if not query.strip():
            raise RetrievalError("Query cannot be empty", sys)

        query_entities = self._extract_query_entities(query)
        logger.info(f"Query entities found: {query_entities}")

        if not query_entities:
            logger.info("No entities found in query — graph retrieval cannot proceed")
            return []

        matched_nodes = self._match_entities_to_nodes(query_entities)
        logger.info(f"Matched to graph nodes: {matched_nodes}")

        if not matched_nodes:
            logger.info("No query entities matched any graph node")
            return []

        results: list[GraphRetrievedChunk] = []
        seen_chunks = set()

        for start_node in matched_nodes:
            paths = self._traverse(start_node, hops=hops)

            for entity_path, relation, target_node, edge_data in paths:
                chunk_key = (edge_data["source"], edge_data["page"], edge_data["chunk_text"][:50])
                if chunk_key in seen_chunks:
                    continue
                seen_chunks.add(chunk_key)

                results.append(GraphRetrievedChunk(
                    text        = edge_data["chunk_text"],
                    source      = edge_data["source"],
                    page        = edge_data["page"],
                    entity_path = entity_path,
                    relation    = relation,
                ))

                if len(results) >= n_results:
                    break
            if len(results) >= n_results:
                break

        logger.info(f"Graph retrieval returned {len(results)} chunks")
        return results

    # ── Internal ─────────────────────────────────────────────

    def _extract_query_entities(self, query: str) -> list[str]:
        """Run spaCy NER on the query, keep only relevant entity types."""
        doc = self.nlp(query)
        return [
            ent.text.strip() for ent in doc.ents
            if ent.label_ in RELEVANT_ENTITY_TYPES
        ]

    def _normalize_for_lookup(self, text: str) -> str:
        """
        Apply the same normalization used when building the graph
        (strip leading article, collapse whitespace) so query entities
        match the normalized node names.
        """
        text = re.sub(r"\s+", " ", text.strip())
        words = text.split(" ")
        if words and words[0].lower() in LEADING_ARTICLES:
            words = words[1:]
        return " ".join(words).strip()

    def _match_entities_to_nodes(self, query_entities: list[str]) -> list[str]:
        """
        Match extracted query entities to actual nodes in the graph using
        exact (case-insensitive) matching only, after applying the same
        normalization used when the graph was built.

        Substring matching was deliberately removed — it caused false
        positives (e.g. "GDPR Article 33" matching unrelated node
        "Article 3") that dragged context precision down to ~0.22-0.25
        in testing. Exact match after normalization is stricter but
        every match returned is trustworthy.
        """
        matched = []
        for qe in query_entities:
            normalized = self._normalize_for_lookup(qe)
            node = self._node_lookup.get(normalized.lower())
            if node:
                matched.append(node)

        return list(set(matched))

    def _traverse(
        self,
        start_node: str,
        hops: int,
    ) -> list[tuple[list[str], str, str, dict]]:
        """
        Walk outward from start_node up to `hops` edges away.
        Returns list of (entity_path, relation, target_node, edge_data) tuples.
        """
        if start_node not in self.graph:
            return []

        results = []
        visited = {start_node}
        frontier = [(start_node, [start_node])]

        for _ in range(hops):
            next_frontier = []

            for node, path in frontier:
                for neighbor in self.graph.successors(node):
                    for _, edge_data in self.graph.get_edge_data(node, neighbor).items():
                        results.append((path + [neighbor], edge_data["relation"], neighbor, edge_data))

                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append((neighbor, path + [neighbor]))

            frontier = next_frontier
            if not frontier:
                break

        return results