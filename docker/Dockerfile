FROM python:3.12-slim AS build
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install ".[pgvector]"

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=build /install /usr/local
COPY sample_docs ./sample_docs
COPY eval ./eval
RUN useradd --system --uid 10001 rag
USER rag
EXPOSE 8000
ENTRYPOINT ["python", "-m", "rag_assistant"]
CMD ["serve"]
