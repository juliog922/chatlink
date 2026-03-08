# src/chatlink_bot/ai/rag.py
import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi
from cudara_client import CudaraClient

from .qdrant import qdrant_service

logger = logging.getLogger("RAG")

CUDARA_URL = os.getenv("CUDARA_URL", "http://cudara:8000")
EMBED_MODEL = os.getenv("CUDARA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("CUDARA_RERANK_MODEL", "").strip()
PAIR_SEP = os.getenv("CUDARA_RERANK_PAIR_SEP", "␞")

RAG_RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true"
RAG_RERANK_POOL = int(os.getenv("RAG_RERANK_POOL", "48"))
RAG_RERANK_MAX_CHARS = int(os.getenv("RAG_RERANK_MAX_CHARS", "1200"))
RAG_RERANK_MAX_LENGTH = int(os.getenv("RAG_RERANK_MAX_LENGTH", "512"))

# Parámetros de selección robusta
FLAT_SCORE_EPS = float(os.getenv("RAG_RERANK_FLAT_EPS", "0.20"))
KEEP_WITHIN_NORM = float(os.getenv("RAG_RERANK_KEEP_WITHIN_NORM", "0.18"))
MIN_KEEP = int(os.getenv("RAG_RERANK_MIN_KEEP", "2"))


class HybridRetriever:
    """
    RAG Híbrido Robusto:
      1) Exact Code Match (Short-circuit) -> Prioridad Absoluta
      2) Dense (Qdrant)
      3) Sparse (BM25)
      4) RRF Fusion
      5) Rerank Remoto (Cudara)
    """

    DENSE_MULT = 12
    SPARSE_MULT = 12
    RRF_K = 60

    _re_nonword = re.compile(r"[^\w]+", re.UNICODE)
    _re_collapse = re.compile(r"\s+", re.UNICODE)

    def __init__(self) -> None:
        self.bm25: Optional[BM25Okapi] = None
        self.documents_map: Dict[str, Dict[str, Any]] = {}
        self.corpus_ids: List[str] = []
        self._cudara = CudaraClient(CUDARA_URL)

    def _normalize_text(self, s: str) -> str:
        s = (s or "").lower()
        s = s.replace("/", " ").replace("\\", " ").replace("_", " ").replace("-", " ")
        s = self._re_nonword.sub(" ", s)
        s = self._re_collapse.sub(" ", s).strip()
        return s

    def _tokenize(self, s: str) -> List[str]:
        s = self._normalize_text(s)
        toks = s.split(" ") if s else []
        return [t for t in toks if 3 <= len(t) <= 40]

    @staticmethod
    def _group_key(doc: Dict[str, Any], point_id: str) -> str:
        return str(doc.get("CodigoArticulo") or doc.get("id") or point_id)

    async def initialize(self) -> None:
        docs = await qdrant_service.get_all_documents()
        if not docs:
            logger.warning("No documents in Qdrant. RAG will be empty.")
            return

        tokenized: List[List[str]] = []
        self.documents_map.clear()
        self.corpus_ids.clear()

        for doc in docs:
            pid = str(doc.get("id") or "")
            content = (doc.get("content") or "").strip()
            if not pid or not content:
                continue
            self.documents_map[pid] = doc
            self.corpus_ids.append(pid)
            tokenized.append(self._tokenize(content))

        self.bm25 = await asyncio.to_thread(BM25Okapi, tokenized)
        logger.info(f"HybridRetriever ready: {len(self.corpus_ids)} docs")

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.bm25:
            return []
        tq = self._tokenize(query)
        if not tq:
            return []
        scores = self.bm25.get_scores(tq)
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        out: List[Tuple[str, float]] = []
        for i in ranked:
            sc = float(scores[i])
            if sc > 0.0:
                out.append((self.corpus_ids[i], sc))
        return out

    @staticmethod
    def _rrf(dense_hits: List[Any], sparse_hits: List[Tuple[str, float]], k: int = 60) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = {}
        for rank, hit in enumerate(dense_hits, 1):
            doc_id = str(hit.id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        for rank, (doc_id, _) in enumerate(sparse_hits, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _minmax01(xs: List[float]) -> List[float]:
        if not xs:
            return []
        mn = min(xs)
        mx = max(xs)
        if (mx - mn) < 1e-9:
            return [0.0 for _ in xs]
        return [(x - mn) / (mx - mn) for x in xs]

    def _pick_topk_with_diversity(self, ranked_ids: List[str], top_k: int) -> List[Dict[str, Any]]:
        final: List[Dict[str, Any]] = []
        seen_groups: set[str] = set()

        for doc_id in ranked_ids:
            doc = self.documents_map.get(doc_id)
            if not doc:
                continue
            gk = self._group_key(doc, doc_id)
            if gk in seen_groups:
                continue
            seen_groups.add(gk)
            final.append(dict(doc))
            if len(final) >= top_k:
                break

        return final

    async def _embed_query(self, query: str) -> List[float]:
        resp = await asyncio.to_thread(self._cudara.embed, EMBED_MODEL, [query])
        if hasattr(resp, "embeddings") and resp.embeddings:
            return resp.embeddings[0]
        if hasattr(resp, "embedding") and resp.embedding:
            return resp.embedding
        return []

    async def _rerank_scores(self, query: str, doc_ids: List[str]) -> Tuple[List[str], List[float]]:
        """
        Obtiene scores del modelo Reranker (Cudara).
        """
        if not RERANK_MODEL or not doc_ids:
            return ([], [])

        packed: List[str] = []
        kept_ids: List[str] = []

        for did in doc_ids:
            doc = self.documents_map.get(did)
            if not doc:
                continue
            text = (doc.get("content") or "").strip()
            if not text:
                continue
            if RAG_RERANK_MAX_CHARS > 0 and len(text) > RAG_RERANK_MAX_CHARS:
                text = text[:RAG_RERANK_MAX_CHARS].rstrip() + "…"
            packed.append(f"{query}{PAIR_SEP}{text}")
            kept_ids.append(did)

        if not packed:
            return ([], [])

        # Llamada a Cudara (endpoint embeddings actuando como reranker)
        resp = await asyncio.to_thread(
            self._cudara.embed,
            RERANK_MODEL,
            packed,
            pair_sep=PAIR_SEP,
            max_length=RAG_RERANK_MAX_LENGTH,
        )

        vecs: List[List[float]] = []
        if hasattr(resp, "embeddings") and resp.embeddings:
            vecs = resp.embeddings
        elif hasattr(resp, "embedding") and resp.embedding:
            vecs = [resp.embedding]

        scores: List[float] = []
        for v in vecs[: len(kept_ids)]:
            if not v:
                scores.append(0.0)
            else:
                # El modelo retorna 1D vector [score]
                scores.append(float(v[0]))

        # Ordenar descendente por score
        combined = sorted(zip(kept_ids, scores), key=lambda x: x[1], reverse=True)
        return ([x[0] for x in combined], [x[1] for x in combined])

    async def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # --- 0. PRIORITY: EXACT CODE CHECK (Short-circuit) ---
        # Si la query es exactamente un código existente, devuélvelo directo.
        clean_q = query.strip()
        exact_hits = await qdrant_service.search_by_exact_code(clean_q)
        if exact_hits:
            results = []
            for point in exact_hits:
                if not point.payload: continue
                doc = dict(point.payload)
                doc["id"] = str(point.id)
                # Forzamos un score altísimo para indicar certeza absoluta
                doc["relevance_score"] = 2.0 
                results.append(doc)
            # Retornamos inmediato, ahorrando BM25/Vector/Rerank
            return results[:top_k]

        # --- Flujo Normal (Vectorial/Híbrido) ---
        if not self.bm25:
            await self.initialize()
            if not self.bm25:
                return []

        # 1. Búsqueda Densa
        qvec = await self._embed_query(query)
        if not qvec:
            return []

        dense_limit = max(top_k * self.DENSE_MULT, top_k * 3)
        dense_hits = await qdrant_service.search_dense(qvec, limit=dense_limit)
        
        # 2. Búsqueda Esparsa
        sparse_limit = max(top_k * self.SPARSE_MULT, top_k * 3)
        loop = asyncio.get_running_loop()
        sparse_hits = await loop.run_in_executor(None, self._bm25_search, query, sparse_limit)

        # 3. Fusión RRF
        fused = self._rrf(dense_hits, sparse_hits, k=self.RRF_K)
        if not fused:
            return []
        
        fused_ids = [doc_id for (doc_id, _) in fused]

        # 4. Reranking y Filtrado
        do_rerank = bool(RAG_RERANK_ENABLED and RERANK_MODEL)
        
        if do_rerank:
            pool_ids = fused_ids[: min(RAG_RERANK_POOL, len(fused_ids))]
            try:
                ordered_ids, ordered_scores = await self._rerank_scores(query, pool_ids)
            except Exception as e:
                logger.warning(f"Rerank failed: {e}")
                ordered_ids, ordered_scores = ([], [])

            if ordered_ids and ordered_scores:
                score_range = max(ordered_scores) - min(ordered_scores)

                # A. Scores Planos -> El reranker está confundido. 
                # Estrategia: Preferir Intersección (Dense AND Sparse) y mantener orden Fused.
                if score_range < FLAT_SCORE_EPS:
                    dense_set = {str(h.id) for h in dense_hits[: max(top_k * 4, top_k)]}
                    sparse_set = {doc_id for doc_id, _ in sparse_hits[: max(top_k * 4, top_k)]}
                    both = dense_set.intersection(sparse_set)

                    # Prioridad: Estar en ambas listas, luego seguir orden RRF
                    prefer = [d for d in fused_ids if d in both]
                    rest = [d for d in fused_ids if d not in both]
                    
                    # Eliminar duplicados preservando orden
                    final_candidates = []
                    seen = set()
                    for x in prefer + rest:
                        if x not in seen:
                            final_candidates.append(x)
                            seen.add(x)

                    final_docs = self._pick_topk_with_diversity(final_candidates, top_k)
                    # Marcar score 0 para indicar incertidumbre
                    for d in final_docs: d["relevance_score"] = 0.0
                    return final_docs

                # B. Scores Diferenciados -> Normalizar y Filtrar
                norm = self._minmax01(ordered_scores)
                top_norm = norm[0] if norm else 0.0
                thr = max(0.0, top_norm - KEEP_WITHIN_NORM)

                kept = [did for did, n in zip(ordered_ids, norm) if float(n) >= thr]
                
                # Asegurar un mínimo de resultados
                if len(kept) < min(MIN_KEEP, top_k):
                    kept = ordered_ids[: max(MIN_KEEP, min(top_k, len(ordered_ids)))]

                final_docs = self._pick_topk_with_diversity(kept, top_k)

                # Asignar score normalizado
                norm_map = {did: float(n) for did, n in zip(ordered_ids, norm)}
                for d in final_docs:
                    pid = str(d.get("id") or "")
                    d["relevance_score"] = float(norm_map.get(pid, 0.0))
                
                return final_docs

        # Fallback: Sin Reranker -> Usar RRF directo
        final: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        
        for doc_id, score in fused:
            if doc_id in seen_ids: continue
            doc = self.documents_map.get(doc_id)
            if not doc: continue
            
            seen_ids.add(doc_id)
            out = dict(doc)
            out["relevance_score"] = float(score)
            final.append(out)
            if len(final) >= top_k: break

        return final


rag_service = HybridRetriever()