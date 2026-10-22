"""Exact frozen BGE encoder and streaming FAISS kernels, loaded on demand."""

from pathlib import Path
import numpy as np

from .vendor.src.embedders.text_embedder import TextEmbedder
from .vendor.src.indexes.streaming_flat_index import StreamingFlatIPIndex
from .offsets import CompatibleOffsets


def embedding_source(resources):
    build = Path(resources["dense"]["bge"]["build"])
    return (
        build / "embeddings" if resources["corpus"]["dataset"] == "hotpotqa" else build
    )


class Retriever:
    def __init__(self, resources, device="cpu"):
        self.resources = resources
        self.device = device
        self.index = self.encoder = self.chunks = None

    def load(self):
        if self.index is not None:
            return
        import torch, faiss

        corpus = self.resources["corpus"]
        spec = self.resources["dense"]["bge"]
        torch.set_num_threads(4)
        faiss.omp_set_num_threads(4)
        self.index = StreamingFlatIPIndex(
            embedding_source(self.resources), corpus_chunk_size=50000
        )
        if self.index.count != corpus["documents"] or self.index.dimension != 384:
            raise ValueError("Incomplete index")
        self.encoder = TextEmbedder(
            backend="sentence_transformers",
            model_name=spec["query_model"],
            revision=spec["query_revision"],
            normalize=True,
            batch_size=16,
            query_prefix="",
            document_prefix="",
            max_sequence_length=512,
            local_files_only=True,
            device=self.device,
        )
        if self.encoder.embedding_space("inner_product").to_dict() != spec["space"]:
            raise ValueError("BGE space changed")
        self.chunks = CompatibleOffsets(
            corpus["chunks"], corpus["offsets"], expected_rows=corpus["documents"]
        )

    def __call__(self, qid, query):
        self.load()
        vectors = np.ascontiguousarray(
            self.encoder.encode_queries([query]), dtype=np.float32
        )
        tokenizer = self.encoder._model[0].tokenizer
        count = len(
            tokenizer(query, add_special_tokens=True, truncation=False)["input_ids"]
        )
        scores, ids = self.index.search_many(vectors, 11, query_batch_size=1)
        docs = []
        for score, chunk in zip(
            scores.tolist()[0], self.chunks.get_many(ids.tolist()[0])
        ):
            if chunk.doc_id == qid:
                continue
            rank = len(docs) + 1
            docs.append(
                {
                    "rank": rank,
                    "score": score,
                    "doc_id": chunk.doc_id,
                    "vector_id": chunk.vector_id,
                    "source": chunk.source,
                    "text": chunk.text,
                    "block": f"[Chunk {rank} | source={chunk.source}, pages={chunk.page_start}-{chunk.page_end}]\n{chunk.text}",
                }
            )
            if len(docs) == 10:
                break
        assert len(docs) == 10
        return {
            "qid": qid,
            "query": query,
            "k": 10,
            "docs": docs,
            "query_tokens": count,
            "encoder_truncated": count > 512,
            "batch_queries": 1,
            "execution": "single-query encoding and FAISS kernels; shared read-only corpus blocks",
        }

    def close(self):
        if self.chunks is not None:
            self.chunks.close()
        if self.index is not None:
            self.index.close()
        self.encoder = None
