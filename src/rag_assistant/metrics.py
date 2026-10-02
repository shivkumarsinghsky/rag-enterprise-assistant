from prometheus_client import Counter, Histogram

QUERIES = Counter("rag_queries_total", "Questions answered", ["grounded"])
STAGE_SECONDS = Histogram(
    "rag_stage_seconds",
    "Latency per pipeline stage",
    ["stage"],
    buckets=(0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
TOKENS = Counter("rag_llm_tokens_total", "LLM tokens used", ["kind"])
FLAGGED = Counter("rag_flagged_chunks_total", "Retrieved chunks excluded by injection heuristics")
INGESTED_CHUNKS = Counter("rag_ingested_chunks_total", "Chunks embedded and stored")
