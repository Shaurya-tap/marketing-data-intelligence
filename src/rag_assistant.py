"""
rag_assistant.py
----------------
The AI ASSISTANT that answers product questions, using RAG.

WHAT IS RAG (Retrieval-Augmented Generation)?
  Imagine an open-book exam. Instead of forcing the language model to "just know"
  everything, we:
    1. RETRIEVE the most relevant product info from our data (the "open book"),
    2. then ask the language model to GENERATE an answer using ONLY that info.
  This keeps answers grounded in real data and reduces made-up ("hallucinated")
  facts.

PIECES USED (all open source):
  - sentence-transformers  -> turns text into vectors (lists of numbers) so we
                              can measure which products are most similar to a
                              question. This is the "embedding" model.
  - a small open LLM (FLAN-T5) via the `transformers` library -> writes the final
                              answer in natural language.
  - numpy cosine similarity -> our lightweight "vector search". (For large data
                              you'd swap this for FAISS; see README.)

The class is built so that if the LLM can't be loaded (e.g. no internet to
download it), it still returns a useful answer from the retrieved text.
"""

import os
import numpy as np
import pandas as pd

from data_prep import load_and_clean

# These heavy imports are done lazily inside the class so that simply importing
# this file (e.g. for the predictor) does not require the ML libraries.


class RAGAssistant:
    def __init__(self, csv_path, embed_model="all-MiniLM-L6-v2",
                 llm_model="google/flan-t5-base"):
        self.csv_path = csv_path
        self.embed_model_name = embed_model
        self.llm_model_name = llm_model

        self.df = load_and_clean(csv_path)
        # Build one text "document" per product — this is our knowledge base.
        self.documents = self._build_documents(self.df)

        self.embedder = None
        self.doc_vectors = None
        self.llm = None
        self._ready = False

    # ---------- building the knowledge base ----------
    def _build_documents(self, df):
        docs = []
        for _, row in df.iterrows():
            text = (
                f"Product: {row['product_name']}. "
                f"Category: {row['main_category']}. "
                f"Actual price: {row['actual_price']}. "
                f"Discounted price: {row['discounted_price']}. "
                f"Discount: {row['discount_percentage']}. "
                f"Rating: {row['rating']} from {row['rating_count']} ratings. "
                f"Description: {row['about_product']}"
            )
            docs.append(text)
        return docs

    # ---------- loading models ----------
    def load(self):
        """Load the embedding model + LLM and pre-compute document vectors."""
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {self.embed_model_name} ...")
        self.embedder = SentenceTransformer(self.embed_model_name)

        print("Embedding the product knowledge base (one-time)...")
        self.doc_vectors = self.embedder.encode(
            self.documents, normalize_embeddings=True, show_progress_bar=False
        )

        # Try to load the open-source LLM. If it fails, we degrade gracefully.
        try:
            from transformers import pipeline
            print(f"Loading LLM: {self.llm_model_name} ...")
            self.llm = pipeline("text2text-generation", model=self.llm_model_name)
        except Exception as e:
            print(f"[warn] Could not load LLM ({e}). "
                  f"Falling back to retrieval-only answers.")
            self.llm = None

        self._ready = True

    # ---------- the RAG steps ----------
    def retrieve(self, question, k=3):
        """Step 1: find the k most relevant product documents for the question."""
        q_vec = self.embedder.encode([question], normalize_embeddings=True)[0]
        # cosine similarity = dot product because vectors are normalized
        scores = self.doc_vectors @ q_vec
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i]), self.documents[i]) for i in top_idx]

    def answer(self, question, k=3):
        """Step 2: generate a grounded answer using the retrieved context."""
        if not self._ready:
            self.load()

        retrieved = self.retrieve(question, k=k)
        context = "\n".join(f"- {doc}" for _, _, doc in retrieved)

        if self.llm is not None:
            prompt = (
                "You are a helpful e-commerce shopping assistant. "
                "Answer the question using ONLY the product context below. "
                "If the answer is not in the context, say you don't have that info.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                "Answer:"
            )
            out = self.llm(prompt, max_new_tokens=128)[0]["generated_text"]
            answer_text = out.strip()
        else:
            # Fallback: return the single best-matching product description.
            answer_text = ("Based on the most relevant product I found:\n"
                           + retrieved[0][2])

        # We always return the sources so the answer is transparent/auditable.
        sources = [{"score": round(s, 3), "text": d} for _, s, d in retrieved]
        return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "data", "amazon_sample.csv")
    bot = RAGAssistant(path)
    result = bot.answer("Which headphones are good for calls and have noise isolation?")
    print("\nANSWER:\n", result["answer"])
    print("\nTOP SOURCE:\n", result["sources"][0]["text"])
