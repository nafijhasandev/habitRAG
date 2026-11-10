"""Integration test from a ten-page PDF excerpt to both search indexes."""

import pymupdf as fitz

import config
from core.bm25_index import BM25Index
from core.vector_store import VectorStore
from ingestion import pipeline


class _FakeEmbeddingEngine:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_pdf_ingestion_populates_matching_chroma_and_bm25_indexes(
    tmp_path, monkeypatch
):
    source_pdf = config.PDF_PATH
    excerpt = tmp_path / "first_ten_pages.pdf"
    with fitz.open(source_pdf) as source, fitz.open() as subset:
        subset.insert_pdf(source, from_page=0, to_page=9)
        subset.save(excerpt)

    monkeypatch.setattr(config, "PDF_PATH", excerpt)
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "BM25_INDEX_PATH", tmp_path / "bm25.pkl")
    monkeypatch.setattr(config, "CHUNKS_JSON_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr("core.embeddings.EmbeddingEngine", _FakeEmbeddingEngine)

    pipeline.run_ingestion()

    store = VectorStore()
    bm25 = BM25Index.load(config.BM25_INDEX_PATH)
    chroma_ids = set(store.collection.get()["ids"])
    bm25_ids = {f"chunk_{chunk.chunk_id}" for chunk in bm25.chunks}
    assert chroma_ids
    assert chroma_ids == bm25_ids
    assert config.CHUNKS_JSON_PATH.exists()
