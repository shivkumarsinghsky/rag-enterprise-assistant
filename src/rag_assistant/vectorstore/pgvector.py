"""PostgreSQL + pgvector store with full-text keyword search.

Tenant and ACL filters are SQL predicates evaluated together with the ANN/FTS search, and every query runs with
the tenant in the WHERE clause. An HNSW index serves cosine distance; a GIN index serves full-text search.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from rag_assistant.embeddings import Vector, tokenize
from rag_assistant.models import Chunk, Filters

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rag_chunks (
  id          text PRIMARY KEY,
  tenant      text NOT NULL,
  document_id text NOT NULL,
  title       text NOT NULL,
  section     text NOT NULL,
  text        text NOT NULL,
  ordinal     int  NOT NULL,
  source      text NOT NULL,
  acl         text[] NOT NULL DEFAULT '{}',
  metadata    jsonb NOT NULL DEFAULT '{}',
  embedding   vector({dims}) NOT NULL,
  tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || section || ' ' || text)) STORED
);
CREATE INDEX IF NOT EXISTS rag_chunks_tenant_doc ON rag_chunks (tenant, document_id);
CREATE INDEX IF NOT EXISTS rag_chunks_embedding ON rag_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS rag_chunks_tsv ON rag_chunks USING gin (tsv);
"""

FILTER_SQL = """
  tenant = %(tenant)s
  AND (cardinality(acl) = 0 OR acl && %(groups)s::text[])
  AND (%(doc_ids)s::text[] IS NULL OR document_id = ANY(%(doc_ids)s::text[]))
  AND metadata @> %(metadata)s::jsonb
"""


def _vec(v: Vector) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in np.asarray(v, dtype=np.float32)) + "]"


class PgVectorStore:
    def __init__(self, dsn: str, dimensions: int) -> None:
        import psycopg

        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute(SCHEMA.replace("{dims}", str(int(dimensions))))

    def _params(self, filters: Filters) -> dict[str, Any]:
        return {
            "tenant": filters.tenant,
            "groups": sorted(filters.groups),
            "doc_ids": sorted(filters.document_ids) if filters.document_ids is not None else None,
            "metadata": json.dumps(filters.metadata),
        }

    @staticmethod
    def _row_to_chunk(row: tuple[Any, ...]) -> Chunk:
        cid, tenant, doc_id, title, section, text, ordinal, source, acl, metadata = row[:10]
        return Chunk(cid, doc_id, tenant, title, section, text, ordinal, source, frozenset(acl), metadata)

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO rag_chunks
                  (id, tenant, document_id, title, section, text, ordinal, source, acl, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, section = EXCLUDED.section, title = EXCLUDED.title,
                  acl = EXCLUDED.acl, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding
                """,
                [
                    (
                        c.id,
                        c.tenant,
                        c.document_id,
                        c.title,
                        c.section,
                        c.text,
                        c.ordinal,
                        c.source,
                        sorted(c.acl),
                        json.dumps(c.metadata),
                        _vec(v),
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def delete_document(self, tenant: str, document_id: str) -> int:
        cur = self._conn.execute("DELETE FROM rag_chunks WHERE tenant = %s AND document_id = %s", (tenant, document_id))
        return int(cur.rowcount)

    def vector_search(self, vector: Vector, filters: Filters, k: int) -> list[tuple[Chunk, float]]:
        rows = self._conn.execute(
            f"""SELECT id, tenant, document_id, title, section, text, ordinal, source, acl, metadata,
                       1 - (embedding <=> %(vec)s::vector) AS score
                  FROM rag_chunks WHERE {FILTER_SQL}
                 ORDER BY embedding <=> %(vec)s::vector LIMIT %(k)s""",
            {**self._params(filters), "vec": _vec(vector), "k": k},
        ).fetchall()
        return [(self._row_to_chunk(r), float(r[10])) for r in rows]

    def keyword_search(self, query: str, filters: Filters, k: int) -> list[tuple[Chunk, float]]:
        # OR semantics (any query term), ranked by ts_rank_cd; terms reduced to [a-z0-9] so input cannot inject
        # tsquery operators.
        terms = sorted({t for tok in tokenize(query) for t in re.split(r"[^a-z0-9]+", tok) if t})
        if not terms:
            return []
        rows = self._conn.execute(
            f"""SELECT id, tenant, document_id, title, section, text, ordinal, source, acl, metadata,
                       ts_rank_cd(tsv, q) AS score
                  FROM rag_chunks, to_tsquery('english', %(q)s) q
                 WHERE {FILTER_SQL} AND tsv @@ q
                 ORDER BY score DESC LIMIT %(k)s""",
            {**self._params(filters), "q": " | ".join(terms), "k": k},
        ).fetchall()
        return [(self._row_to_chunk(r), float(r[10])) for r in rows]

    def documents(self, tenant: str) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT document_id, min(title), min(source), count(*) FROM rag_chunks WHERE tenant = %s"
            " GROUP BY document_id ORDER BY document_id",
            (tenant,),
        ).fetchall()
        return [{"id": r[0], "title": r[1], "source": r[2], "chunks": r[3]} for r in rows]
