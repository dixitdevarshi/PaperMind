import re
import sys
import pickle
from pathlib import Path

import spacy
import networkx as nx

from src.logger import get_logger
from src.exception import PaperMindException
from src.components.embedding_engine import EmbeddingEngine

logger = get_logger(__name__)

GRAPH_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "knowledge_graph.gpickle"

# Entity types worth keeping as graph nodes — skip noise like CARDINAL, ORDINAL
RELEVANT_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "LAW", "NORP", "PRODUCT", "EVENT", "FAC", "WORK_OF_ART"
}

# Leading articles/determiners to strip during normalization, so
# "the Volkswagen Group" and "Volkswagen Group" merge into one node
LEADING_ARTICLES = {"the", "a", "an", "der", "die", "das", "den", "dem"}

# Relation label used to connect every entity in a document back to
# that document's own node — lets coarse queries like "GDPR" or "Bosch"
# (the document/company name itself) find something in the graph even
# when no specific sub-entity is mentioned in the query
DOC_NODE_RELATION = "appears_in"


class GraphBuilder:
    """
    Extracts entities and relationships from document chunks and builds
    a NetworkX graph. Nodes are entities, edges are relationships, both
    tagged with source document and chunk text for traceability.

    Two structural additions beyond plain co-occurrence:
      1. Entity text is normalized (case + leading article stripped)
         before being used as a node key, so near-duplicate mentions of
         the same entity collapse into a single node.
      2. Every document gets its own node (named after its source file,
         with the .pdf extension and underscores stripped for readability),
         connected to every entity found within it. This lets a query like
         "What does the GDPR say" find something even without a specific
         article number or named entity.

    Usage:
        builder = GraphBuilder()
        graph   = builder.build_from_chunks(chunks)
        builder.save(graph)
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        logger.info(f"Loading spaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except Exception as e:
            raise PaperMindException(
                f"Failed to load spaCy model '{model_name}'. "
                f"Run: python -m spacy download {model_name}", sys
            )
        logger.info("GraphBuilder ready")

    def build_from_chunks(self, chunks: list[dict]) -> nx.MultiDiGraph:
        """
        Build a knowledge graph from a list of chunk dicts.
        Each chunk dict needs keys: text, source, page.

        Returns a NetworkX MultiDiGraph where:
          - nodes = normalized entities + one node per source document
          - edges = co-occurrence relationships, tagged with source chunk
        """
        graph = nx.MultiDiGraph()
        skipped_self_loops = 0
        skipped_garbage     = 0

        for i, chunk in enumerate(chunks):
            text   = chunk["text"]
            source = chunk.get("source", "unknown")
            page   = chunk.get("page", "?")

            triples = self._extract_triples(text)

            for entity_a, relation, entity_b in triples:
                result = self._add_triple(
                    graph=graph,
                    entity_a=entity_a,
                    relation=relation,
                    entity_b=entity_b,
                    source=source,
                    page=page,
                    chunk_text=text,
                )
                if result == "self_loop":
                    skipped_self_loops += 1
                elif result == "garbage":
                    skipped_garbage += 1

            # Connect every entity mentioned in this chunk to the document node
            self._link_entities_to_document(graph, text, source, page)

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(chunks)} chunks")

        logger.info(
            f"Graph built — {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges "
            f"(skipped {skipped_self_loops} self-loops, {skipped_garbage} garbage entities)"
        )
        return graph

    def add_chunks_to_existing_graph(
        self,
        chunks: list[dict],
        path: str | None = None,
    ) -> nx.MultiDiGraph:
        """
        Incrementally update an existing graph with new chunks, instead of
        rebuilding from scratch. Used when a new document is ingested —
        only processes the new chunks, then merges into the saved graph.

        If no graph exists yet on disk, creates a new one.

        Parameters:
            chunks : list of chunk dicts (text, source, page) — typically
                     from a single newly-ingested document
            path   : optional override for the graph file location

        Returns the updated graph (also saved to disk).
        """
        load_path = Path(path) if path else GRAPH_PATH

        if load_path.exists():
            graph = self.load(str(load_path))
            logger.info(f"Loaded existing graph — adding {len(chunks)} new chunks")
        else:
            graph = nx.MultiDiGraph()
            logger.info(f"No existing graph found — creating new one with {len(chunks)} chunks")

        skipped_self_loops = 0
        skipped_garbage     = 0

        for chunk in chunks:
            text   = chunk["text"]
            source = chunk.get("source", "unknown")
            page   = chunk.get("page", "?")

            triples = self._extract_triples(text)

            for entity_a, relation, entity_b in triples:
                result = self._add_triple(
                    graph=graph,
                    entity_a=entity_a,
                    relation=relation,
                    entity_b=entity_b,
                    source=source,
                    page=page,
                    chunk_text=text,
                )
                if result == "self_loop":
                    skipped_self_loops += 1
                elif result == "garbage":
                    skipped_garbage += 1

            self._link_entities_to_document(graph, text, source, page)

        logger.info(
            f"Graph updated — now {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges "
            f"(skipped {skipped_self_loops} self-loops, {skipped_garbage} garbage entities)"
        )

        self.save(graph, str(load_path))
        return graph

    def save(self, graph: nx.MultiDiGraph, path: str | None = None) -> None:
        """Save graph to disk as a pickle file."""
        save_path = Path(path) if path else GRAPH_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(graph, f)
        logger.info(f"Graph saved to: {save_path}")

    def load(self, path: str | None = None) -> nx.MultiDiGraph:
        """Load a previously saved graph from disk."""
        load_path = Path(path) if path else GRAPH_PATH
        if not load_path.exists():
            raise PaperMindException(f"No graph found at: {load_path}", sys)
        with open(load_path, "rb") as f:
            graph = pickle.load(f)
        logger.info(
            f"Graph loaded — {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        return graph

    # ── Internal ─────────────────────────────────────────────

    def _normalize_entity(self, text: str) -> str:
        """
        Normalize entity text so near-duplicate mentions collapse into
        one node. Strips a single leading article/determiner and collapses
        internal whitespace, but otherwise preserves the original casing
        of the remaining text for readability when printed.

        Examples:
          "the Volkswagen Group"   -> "Volkswagen Group"
          "THE VOLKSWAGEN GROUP"   -> "VOLKSWAGEN GROUP"
          "der Verantwortliche"    -> "Verantwortliche"
        """
        text = re.sub(r"\s+", " ", text.strip())

        words = text.split(" ")
        if words and words[0].lower() in LEADING_ARTICLES:
            words = words[1:]

        return " ".join(words).strip()

    def _document_node_name(self, source: str) -> str:
        """
        Convert a source filename into a clean document node name.
        'GDPR_EN.pdf' -> 'GDPR EN', 'VW_Annual_Report_2023.pdf' -> 'VW Annual Report 2023'
        """
        name = source
        if name.lower().endswith(".pdf"):
            name = name[:-4]
        return name.replace("_", " ").strip()

    def _link_entities_to_document(
        self,
        graph: nx.MultiDiGraph,
        text: str,
        source: str,
        page,
    ) -> None:
        """
        Add an edge from the document's own node to every entity found
        in this chunk, labeled 'appears_in'. This lets a query that only
        names the document or company (e.g. "GDPR", "Bosch") find a
        starting point in the graph even with no specific sub-entity.
        """
        doc       = self.nlp(text)
        doc_node  = self._document_node_name(source)
        graph.add_node(doc_node)

        seen_in_chunk = set()
        for ent in doc.ents:
            if ent.label_ not in RELEVANT_ENTITY_TYPES:
                continue

            entity = self._normalize_entity(ent.text)
            if not entity or entity.endswith("-"):
                continue
            if entity.lower() == doc_node.lower():
                continue
            if entity.lower() in seen_in_chunk:
                continue
            seen_in_chunk.add(entity.lower())

            graph.add_node(entity)
            graph.add_edge(
                doc_node,
                entity,
                relation=DOC_NODE_RELATION,
                source=source,
                page=page,
                chunk_text=text[:200],
            )

    def _extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        """
        Extract (entity, relationship, entity) triples from a chunk of text.

        Approach: find all named entities in the text, normalize each one,
        then for every pair of entities that appear in the same sentence,
        create a co-occurrence relationship. The relation label comes from
        the verb connecting them if spaCy's dependency parse finds one,
        otherwise "related_to".
        """
        doc     = self.nlp(text)
        triples = []

        for sent in doc.sents:
            entities = [
                ent for ent in sent.ents
                if ent.label_ in RELEVANT_ENTITY_TYPES
            ]

            if len(entities) < 2:
                continue

            for idx_a in range(len(entities)):
                for idx_b in range(idx_a + 1, len(entities)):
                    ent_a = entities[idx_a]
                    ent_b = entities[idx_b]

                    relation  = self._find_relation(sent, ent_a, ent_b)
                    norm_a    = self._normalize_entity(ent_a.text)
                    norm_b    = self._normalize_entity(ent_b.text)
                    triples.append((norm_a, relation, norm_b))

        return triples

    def _find_relation(self, sent, ent_a, ent_b) -> str:
        """
        Try to find a verb connecting two entities using dependency parsing.
        Falls back to "related_to" if no clear verb is found.
        """
        for token in sent:
            if token.pos_ == "VERB":
                subtree_span = (
                    min(t.i for t in token.subtree),
                    max(t.i for t in token.subtree),
                )
                if (
                    subtree_span[0] <= ent_a.start and ent_a.end <= subtree_span[1] + 1
                ) or (
                    subtree_span[0] <= ent_b.start and ent_b.end <= subtree_span[1] + 1
                ):
                    return token.lemma_

        return "related_to"

    def _add_triple(
        self,
        graph: nx.MultiDiGraph,
        entity_a: str,
        relation: str,
        entity_b: str,
        source: str,
        page,
        chunk_text: str,
    ) -> str | None:
        """
        Add a triple to the graph, creating nodes if they don't exist.
        Filters out self-loops and hyphenation artifacts from PDF extraction.
        Entities are expected to already be normalized by the caller.

        Returns "self_loop" or "garbage" if skipped, None if added successfully.
        """
        if not entity_a or not entity_b:
            return "garbage"

        if entity_a.lower() == entity_b.lower():
            return "self_loop"

        if entity_a.endswith("-") or entity_b.endswith("-"):
            return "garbage"

        graph.add_node(entity_a)
        graph.add_node(entity_b)
        graph.add_edge(
            entity_a,
            entity_b,
            relation=relation,
            source=source,
            page=page,
            chunk_text=chunk_text[:200],
        )
        return None


if __name__ == "__main__":
    # Standalone test — build graph from everything currently in ChromaDB
    from dotenv import load_dotenv
    load_dotenv()

    engine = EmbeddingEngine()
    all_data = engine.collection.get(include=["documents", "metadatas"])

    chunks = [
        {
            "text":   doc,
            "source": meta.get("source", "unknown"),
            "page":   meta.get("page", "?"),
        }
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    print(f"Building graph from {len(chunks)} chunks...")

    builder = GraphBuilder()
    graph   = builder.build_from_chunks(chunks)
    builder.save(graph)

    print(f"\nGraph stats:")
    print(f"  Nodes: {graph.number_of_nodes()}")
    print(f"  Edges: {graph.number_of_edges()}")

    print(f"\nSample nodes:")
    for node in list(graph.nodes())[:10]:
        print(f"  {node}")

    print(f"\nSample edges:")
    for u, v, data in list(graph.edges(data=True))[:10]:
        print(f"  {u} --[{data['relation']}]--> {v}  (from {data['source']}, page {data['page']})")