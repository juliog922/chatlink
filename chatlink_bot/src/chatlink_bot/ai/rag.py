# src/chatlink_bot/ai/rag.py
import asyncio
import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from .cima_client import CimaClient
from .qdrant import qdrant_service

logger = logging.getLogger("RAG")

CIMA_URL = os.getenv("CIMA_URL", "http://cima:8000")
EMBED_MODEL = os.getenv("CIMA_MODEL", "unsloth/gemma-4-E2B-it-GGUF:Q8_0")
# cima has no reranking endpoint, so this stays empty by default and the
# retriever uses its RRF fusion path. Left configurable for a future engine.
RERANK_MODEL = os.getenv("CIMA_RERANK_MODEL", "").strip()

RAG_RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true"
RAG_RERANK_POOL = int(os.getenv("RAG_RERANK_POOL", "48"))
RAG_RERANK_MAX_CHARS = int(os.getenv("RAG_RERANK_MAX_CHARS", "1200"))

# Fusion weights. While gemma mean-pool embedding quality is unproven for this
# catalog, keyword (BM25) evidence outweighs dense similarity: a query
# containing "crema" must surface cream products even if the vector space is
# noisy. Raise the dense weight only after verifying embedding quality.
RAG_DENSE_WEIGHT = float(os.getenv("RAG_DENSE_WEIGHT", "1.0"))
RAG_SPARSE_WEIGHT = float(os.getenv("RAG_SPARSE_WEIGHT", "2.5"))

# Robust-selection parameters (reranker path).
FLAT_SCORE_EPS = float(os.getenv("RAG_RERANK_FLAT_EPS", "0.20"))
KEEP_WITHIN_NORM = float(os.getenv("RAG_RERANK_KEEP_WITHIN_NORM", "0.18"))
MIN_KEEP = int(os.getenv("RAG_RERANK_MIN_KEEP", "2"))

# Stopwords are stripped from BM25 tokens AFTER normalization: "crema PARA LA
# cara" must score on crema/cara. Includes number words ("DOS cremas") and
# request verbs, which are quantity/politeness noise, not product signal.
_SPANISH_STOPWORDS = frozenset(
    "para con una uno unos unas del las los que este esta estos estas "
    "por como mas muy sin sobre entre hasta desde cuando donde quiero "
    "quisiera necesito busco tienes tiene hay algo algun alguna dime cual "
    "cuales dos tres cuatro cinco seis siete ocho nueve diez docena media "
    "quiero ponme apunta apuntame dame mismo misma vez".split()
)

# Product codes in SAGE always contain a digit ("14-1127", "A026075",
# "IN110201BAN", "82301"). Gating the exact-code short-circuit on this shape
# stops every plain-language query ("hola", "crema") from paying a pointless
# Qdrant scroll, and stops words from ever "exact-matching" anything.
_re_code_shape = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-./]{2,20}$")


def _looks_like_product_code(query: str) -> bool:
    q = query.strip()
    return bool(_re_code_shape.match(q)) and any(ch.isdigit() for ch in q)


# Catalog names arrive in Spanish, English, or a mix, and clients speak
# either. This curated ES<->EN cosmetics lexicon expands QUERY tokens at
# match time so BM25 and coverage see through the language: "brocha" matches
# BRUSH, "transparente" matches CLEAR. Deterministic, dependency-free, and
# applied only to queries — the index never changes, embeddings untouched
# (the dense side is already multilingual). Entries are normalized-singular,
# matching _tokenize output.
_BILINGUAL_PAIRS = [
    ("brocha", "brush"), ("pincel", "brush"), ("crema", "cream"),
    ("esmalte", "polish"), ("quitaesmalte", "remover"), ("uña", "nail"),
    ("labio", "lip"), ("labial", "lipstick"), ("pestaña", "lash"),
    ("ceja", "brow"), ("pelo", "hair"), ("cabello", "hair"),
    ("piel", "skin"), ("mano", "hand"), ("pie", "foot"), ("ojo", "eye"),
    ("cara", "face"), ("rostro", "face"), ("cuerpo", "body"),
    ("jabon", "soap"), ("aceite", "oil"), ("polvo", "powder"),
    ("sombra", "shadow"), ("colorete", "blush"), ("rimel", "mascara"),
    ("delineador", "liner"), ("mascarilla", "mask"), ("champu", "shampoo"),
    ("acondicionador", "conditioner"), ("hidratante", "moisturizer"),
    ("limpiador", "cleanser"), ("exfoliante", "scrub"), ("brillo", "gloss"),
    ("mate", "matte"), ("lima", "file"), ("tijera", "scissors"),
    ("pinza", "tweezers"), ("toalla", "towel"), ("algodon", "cotton"),
    ("guante", "glove"), ("secador", "dryer"), ("caja", "box"),
    ("negro", "black"), ("blanco", "white"), ("rojo", "red"),
    ("rosa", "pink"), ("azul", "blue"), ("verde", "green"),
    ("dorado", "gold"), ("plateado", "silver"), ("morado", "purple"),
    ("violeta", "purple"), ("marron", "brown"), ("gris", "grey"),
    ("transparente", "clear"), ("fuerte", "strong"), ("suave", "soft"),
]
_XLATE: Dict[str, set] = {}
for _es, _en in _BILINGUAL_PAIRS:
    _XLATE.setdefault(_es, set()).add(_en)
    _XLATE.setdefault(_en, set()).add(_es)


class HybridRetriever:
    """
    Hybrid RAG:
      1) Exact code match (short-circuit, code-shaped queries only)
      2) Dense (Qdrant) + Sparse (BM25) -> weighted RRF fusion
      3) Optional remote rerank (disabled: cima has no rerank endpoint)

    BM25 tokens are NORMALIZED (lowercase, accents stripped, crude Spanish
    de-pluralization) on BOTH the corpus and the query. Without this, a client
    saying "cremas" scored zero against catalog rows saying "CREMA", and the
    only surviving token "cara" (face) happily matched "DOBLE CARA"
    (double-SIDED) posters — observed live: a face-cream request returned
    nail polish and banners.
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
        self._corpus_token_sets: List[set] = []   # aligned with corpus_ids, for coverage ranking
        self._vocab: set = set()                   # every token in the index
        self._cima = CimaClient(CIMA_URL)

    # ------------------------------------------------------- normalization
    @staticmethod
    def _strip_accents(s: str) -> str:
        """Strip accents but PRESERVE ñ: NFD decomposes ñ into n+tilde, and
        stripping it turns 'uñas' (nails) into 'unas' — a stopword article —
        deleting the word from the query entirely."""
        s = s.replace("ñ", "\x00").replace("Ñ", "\x01")
        s = "".join(c for c in unicodedata.normalize("NFD", s)
                    if unicodedata.category(c) != "Mn")
        return s.replace("\x00", "ñ").replace("\x01", "Ñ")

    @staticmethod
    def _singularize(token: str) -> str:
        """Crude Spanish de-pluralization: cremas->crema, uñas->uña,
        ojos->ojo. Imperfect (lapices->lapice) but applied identically to
        corpus and query, so matching still works within the index."""
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    def _normalize_text(self, s: str) -> str:
        s = self._strip_accents((s or "").lower())
        s = s.replace("/", " ").replace("\\", " ").replace("_", " ").replace("-", " ")
        s = self._re_nonword.sub(" ", s)
        return self._re_collapse.sub(" ", s).strip()

    def _tokenize(self, s: str) -> List[str]:
        tokens = self._normalize_text(s).split(" ") if s else []
        out: List[str] = []
        for t in tokens:
            if not (3 <= len(t) <= 40) or t in _SPANISH_STOPWORDS:
                continue
            t = self._singularize(t)
            if t not in _SPANISH_STOPWORDS:
                out.append(t)
        return out


    def _variants(self, token: str) -> set:
        """The token plus its cross-language translations."""
        return {token} | _XLATE.get(token, set())

    def _known(self, token: str) -> bool:
        """Does ANY variant of this token exist in the index vocabulary?
        Unknown words (typos, chit-chat) are excluded from coverage math so
        they cannot doom an otherwise perfect match to 'ambiguous'."""
        return any(v in self._vocab for v in self._variants(token))

    def _covered(self, token: str, doc_tokens: set) -> bool:
        return any(v in doc_tokens for v in self._variants(token))

    @staticmethod
    def _group_key(doc: Dict[str, Any], point_id: str) -> str:
        return str(doc.get("CodigoArticulo") or doc.get("id") or point_id)

    # ----------------------------------------------------------- indexing
    async def initialize(self) -> None:
        docs = await qdrant_service.get_all_documents()
        if not docs:
            logger.warning("No documents in Qdrant. RAG will be empty.")
            return

        tokenized: List[List[str]] = []
        self.documents_map.clear()
        self.corpus_ids.clear()
        self._corpus_token_sets.clear()
        self._vocab.clear()

        for doc in docs:
            pid = str(doc.get("id") or "")
            content = (doc.get("content") or "").strip()
            if not pid or not content:
                continue
            self.documents_map[pid] = doc
            self.corpus_ids.append(pid)
            # Brand comes from the PAYLOAD too: it sits last in the embedded
            # text and a long description can push it past the char cap — but
            # a client's strongest disambiguator must always be searchable.
            brand_toks = self._tokenize(str(doc.get("MarcaProducto") or ""))
            toks = self._tokenize(content) + [t for t in brand_toks]
            tokenized.append(toks)
            self._corpus_token_sets.append(set(toks))
            self._vocab.update(toks)

        self.bm25 = await asyncio.to_thread(BM25Okapi, tokenized)
        logger.info(f"HybridRetriever ready: {len(self.corpus_ids)} docs")

    # ------------------------------------------------------------ search
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """BM25 candidates re-ranked by (token coverage, head-noun match, score).

        Raw BM25 IDF favors RARE tokens: in 'crema para la cara nivea' the
        common product noun 'crema' is outweighed by the rare qualifier
        'cara', so buffers '4 CARAS' and 'DOBLE CARA' banners beat every
        actual cream (observed live, twice). Coverage first means docs
        matching MORE query terms always win; among single-term matches the
        query's FIRST content token — Spanish is noun-first: '_crema_ para la
        cara' — beats a match on a trailing qualifier."""
        if not self.bm25:
            return []
        tq = self._tokenize(query)
        if not tq:
            return []
        # Cross-language: score with the token set EXPANDED by the lexicon
        # ("brocha" also scores "brush" docs) and count coverage per ORIGINAL
        # token via variants, restricted to tokens the index actually knows.
        expanded = [v for t in tq for v in sorted(self._variants(t))]
        scores = self.bm25.get_scores(expanded)
        known_q = [t for t in tq if self._known(t)]
        head = tq[0]
        candidates = [i for i in range(len(scores)) if float(scores[i]) > 0.0]
        candidates.sort(key=lambda i: (
            -sum(1 for t in known_q if self._covered(t, self._corpus_token_sets[i])),  # 1) terms covered
            -(1 if self._covered(head, self._corpus_token_sets[i]) else 0),            # 2) product noun present
            -float(scores[i]),                                                          # 3) BM25
        ))
        return [(self.corpus_ids[i], float(scores[i])) for i in candidates[:top_k]]

    @staticmethod
    def _rrf(dense_hits: List[Any], sparse_hits: List[Tuple[str, float]], k: int = 60) -> List[Tuple[str, float]]:
        """Weighted RRF: sparse (keyword) evidence outweighs dense similarity
        until the embedder is validated for this catalog."""
        scores: Dict[str, float] = {}
        for rank, hit in enumerate(dense_hits, 1):
            doc_id = str(hit.id)
            scores[doc_id] = scores.get(doc_id, 0.0) + RAG_DENSE_WEIGHT / (k + rank)
        for rank, (doc_id, _) in enumerate(sparse_hits, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + RAG_SPARSE_WEIGHT / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _minmax01(xs: List[float]) -> List[float]:
        if not xs:
            return []
        mn, mx = min(xs), max(xs)
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
        resp = await asyncio.to_thread(self._cima.embed, model=EMBED_MODEL, input=[query])
        embeddings = resp.embeddings or []
        return embeddings[0] if embeddings else []

    async def _rerank_scores(self, query: str, doc_ids: List[str]) -> Tuple[List[str], List[float]]:
        if not RERANK_MODEL or not doc_ids:
            return ([], [])
        input_list, kept_ids = [query], []
        for did in doc_ids:
            doc = self.documents_map.get(did)
            text = ((doc or {}).get("content") or "").strip()
            if not text:
                continue
            if RAG_RERANK_MAX_CHARS > 0 and len(text) > RAG_RERANK_MAX_CHARS:
                text = text[:RAG_RERANK_MAX_CHARS].rstrip() + "…"
            input_list.append(text)
            kept_ids.append(did)
        if len(input_list) <= 1:
            return ([], [])
        resp = await asyncio.to_thread(
            self._cima.embed, model=RERANK_MODEL, input=input_list, options={"is_rerank": True})
        scores = resp.scores or []
        final_scores = [float(scores[i]) if i < len(scores) else 0.0 for i in range(len(kept_ids))]
        combined = sorted(zip(kept_ids, final_scores), key=lambda x: x[1], reverse=True)
        return ([x[0] for x in combined], [x[1] for x in combined])

    async def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # --- 0. Exact code short-circuit, code-shaped queries only ---
        clean_q = query.strip()
        if _looks_like_product_code(clean_q):
            exact_hits = await qdrant_service.search_by_exact_code(clean_q)
            if exact_hits:
                results = []
                for point in exact_hits:
                    if not point.payload:
                        continue
                    doc = dict(point.payload)
                    doc["id"] = str(point.id)
                    doc["relevance_score"] = 2.0   # absolute certainty marker
                    doc["covers_query"] = True
                    doc["query_uncovered"] = []
                    results.append(doc)
                return results[:top_k]

        # --- Hybrid flow ---
        if not self.bm25:
            await self.initialize()
            if not self.bm25:
                return []

        qvec = await self._embed_query(query)
        if not qvec:
            return []

        dense_limit = max(top_k * self.DENSE_MULT, top_k * 3)
        dense_hits = await qdrant_service.search_dense(qvec, limit=dense_limit)

        sparse_limit = max(top_k * self.SPARSE_MULT, top_k * 3)
        loop = asyncio.get_running_loop()
        sparse_hits = await loop.run_in_executor(None, self._bm25_search, query, sparse_limit)

        # A query whose tokens (and translations) are ALL absent from the
        # index vocabulary gets no lexical signal — any sparse hits are pure
        # noise. Fall back to dense-only fusion instead of letting the 2.5x
        # sparse weight amplify garbage (multilingual embeddings carry these).
        if not any(self._known(t) for t in self._tokenize(query)):
            sparse_hits = []

        # Diagnostics: tells apart embedder noise (dense) from keyword misses.
        logger.info(
            f"[RAG] q='{query}' dense={len(dense_hits)} "
            f"top{[(str(h.id), round(float(getattr(h, 'score', 0.0)), 3)) for h in dense_hits[:3]]} | "
            f"sparse={len(sparse_hits)} top{[(d, round(sc, 2)) for d, sc in sparse_hits[:3]]}"
        )

        fused = self._rrf(dense_hits, sparse_hits, k=self.RRF_K)
        if not fused:
            return []
        fused_ids = [doc_id for (doc_id, _) in fused]

        # --- Optional rerank path (inactive without RERANK_MODEL) ---
        if RAG_RERANK_ENABLED and RERANK_MODEL:
            pool_ids = fused_ids[: min(RAG_RERANK_POOL, len(fused_ids))]
            try:
                ordered_ids, ordered_scores = await self._rerank_scores(query, pool_ids)
            except Exception as e:
                logger.warning(f"Rerank failed: {e}")
                ordered_ids, ordered_scores = ([], [])

            if ordered_ids and ordered_scores:
                score_range = max(ordered_scores) - min(ordered_scores)
                if score_range < FLAT_SCORE_EPS:
                    # Flat scores: prefer docs present in BOTH legs, keep RRF order.
                    dense_set = {str(h.id) for h in dense_hits[: max(top_k * 4, top_k)]}
                    sparse_set = {doc_id for doc_id, _ in sparse_hits[: max(top_k * 4, top_k)]}
                    both = dense_set & sparse_set
                    prefer = [d for d in fused_ids if d in both]
                    rest = [d for d in fused_ids if d not in both]
                    final_docs = self._pick_topk_with_diversity(prefer + rest, top_k)
                    for d in final_docs:
                        d["relevance_score"] = 0.0   # uncertainty marker
                    return final_docs

                norm = self._minmax01(ordered_scores)
                thr = max(0.0, (norm[0] if norm else 0.0) - KEEP_WITHIN_NORM)
                kept = [did for did, n in zip(ordered_ids, norm) if float(n) >= thr]
                if len(kept) < min(MIN_KEEP, top_k):
                    kept = ordered_ids[: max(MIN_KEEP, min(top_k, len(ordered_ids)))]
                final_docs = self._pick_topk_with_diversity(kept, top_k)
                norm_map = {did: float(n) for did, n in zip(ordered_ids, norm)}
                for d in final_docs:
                    d["relevance_score"] = float(norm_map.get(str(d.get("id") or ""), 0.0))
                return final_docs

        # --- Default: RRF order with per-code diversity ---
        final_docs = self._pick_topk_with_diversity(fused_ids, top_k)
        score_map = dict(fused)
        q_tokens = list(dict.fromkeys(self._tokenize(query)))
        known_q = [t for t in q_tokens if self._known(t)]
        id_to_idx = {pid: i for i, pid in enumerate(self.corpus_ids)}
        result_sets = [self._corpus_token_sets[id_to_idx[str(d.get("id") or "")]]
                       for d in final_docs if str(d.get("id") or "") in id_to_idx]
        # Tokens the client used that NO returned option contains — across
        # languages. 'crema cara nivea' -> ['nivea'] when no product carries
        # the brand: the agent must say so instead of playing fortune teller.
        uncovered = [t for t in q_tokens
                     if not any(self._covered(t, s) for s in result_sets)]
        for d in final_docs:
            d["query_uncovered"] = uncovered
            d["relevance_score"] = float(score_map.get(str(d.get("id") or ""), 0.0))
            # Full-coverage flag: the doc contains EVERY known content token of
            # the query, across languages ("brocha" covered by BRUSH) — the
            # signal the triage uses to resolve an item without alternatives.
            # Unknown tokens are excluded so a typo can't doom the verdict.
            idx = id_to_idx.get(str(d.get("id") or ""))
            d["covers_query"] = bool(known_q) and idx is not None and \
                all(self._covered(t, self._corpus_token_sets[idx]) for t in known_q)
        return final_docs


rag_service = HybridRetriever()