# src/chatlink_bot/ai/rag.py
import asyncio
import logging
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi
from sqlalchemy import and_, delete, func, select

from .cima_client import CimaClient
from .qdrant import qdrant_service
from ..database import AsyncSessionPG, AsyncSessionSQL
from ..models import (
    ClientHistorySync,
    ClientProductHistory,
    MSArticle,
    MSOrderHeader,
    MSOrderLine,
    es_vendible,
    K_BOT_VENDIBLE,
)
# Stdlib-only module (no ORM, no Qdrant), so importing it here cannot create a
# cycle. It holds the fake order history of simulated clients.
from ..logic.sim_clients import sim_history_cache

logger = logging.getLogger("RAG")
hist_logger = logging.getLogger("ClientHistory")

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


# ---------------------------------------------------------------- normalizing
# These were private methods of HybridRetriever. They are module-level now
# because a SECOND consumer needs them: ClientHistoryRetriever (below) matches
# a client's informal wording against the products that client has actually
# bought, and it MUST normalize exactly the way the index does. Two normalizers
# that drift apart is the worst kind of bug — search still returns *something*.
# One definition; the retriever methods delegate.
_re_nonword = re.compile(r"[^\w]+", re.UNICODE)
_re_collapse = re.compile(r"\s+", re.UNICODE)


def strip_accents(s: str) -> str:
    """Strip accents but PRESERVE ñ: NFD decomposes ñ into n+tilde, and
    stripping it turns 'uñas' (nails) into 'unas' — a stopword article —
    deleting the word from the query entirely."""
    s = s.replace("ñ", "\x00").replace("Ñ", "\x01")
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return s.replace("\x00", "ñ").replace("\x01", "Ñ")


def singularize(token: str) -> str:
    """Crude Spanish de-pluralization: cremas->crema, uñas->uña, ojos->ojo.
    Imperfect (lapices->lapice) but applied identically to corpus and query,
    so matching still works within the index."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_text(s: str) -> str:
    s = strip_accents((s or "").lower())
    s = s.replace("/", " ").replace("\\", " ").replace("_", " ").replace("-", " ")
    s = _re_nonword.sub(" ", s)
    return _re_collapse.sub(" ", s).strip()


def tokenize(s: str) -> List[str]:
    tokens = normalize_text(s).split(" ") if s else []
    out: List[str] = []
    for t in tokens:
        if not (3 <= len(t) <= 40) or t in _SPANISH_STOPWORDS:
            continue
        t = singularize(t)
        if t not in _SPANISH_STOPWORDS:
            out.append(t)
    return out


def coverage_rank(query: str,
                  items: Sequence[Tuple[str, str, str]]) -> List[Tuple[str, int, int]]:
    """Rank candidates by how much of the CLIENT'S wording each one explains.

    THE single ranking rule of this system. It used to exist six times — in the
    retriever, in the BM25 re-rank, in the history matcher, in the match guard —
    and every fix had to be made three times or it was only half applied. One
    function, one behaviour, one place to fix.

    Two numbers, in this order:

      1. COVERED — how many of the client's words the candidate accounts for.
         More is better: a word the client said and the product does not have
         is the strongest possible evidence against it.
      2. EXTRA — how many words the candidate's NAME adds that the client never
         said. Fewer is better: among products that explain everything, the
         tightest is the one that invents least. This is what separates
         "KATIVA KERATINA SHAMPOO 250 ML" from the same thing "PACK PROMO", and
         a mascara from its point-of-sale label.

    `items` is (key, coverage_text, name). The two texts differ on purpose:
    coverage may legitimately be found in the long marketing description, while
    EXTRA must look only at the name — counting description words measured how
    chatty a product sheet was and buried the exact match under four others.

    Equal on both counts means the client's own message cannot separate them,
    and the caller must ask rather than choose.
    """
    q = list(dict.fromkeys(tokenize(query)))
    if not q:
        return [(k, 0, 0) for k, _, _ in items]
    wanted = {v for t in q for v in variants(t)}
    ranked: List[Tuple[str, int, int]] = []
    for key, cover_text, name in items:
        doc = set(tokenize(cover_text or ""))
        covered = sum(1 for t in q if any(v in doc for v in variants(t)))
        extra = len({t for t in tokenize(name or cover_text or "") if t not in wanted})
        ranked.append((key, covered, extra))
    ranked.sort(key=lambda r: (-r[1], r[2]))
    return ranked


def variants(token: str) -> set:
    """The token plus its cross-language translations."""
    return {token} | _XLATE.get(token, set())


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
    # Thin delegations to the module-level functions above. Kept as methods so
    # every existing call site (self._tokenize, self._variants, ...) is
    # unchanged; the rules themselves now live in exactly one place.
    _strip_accents = staticmethod(strip_accents)
    _singularize = staticmethod(singularize)

    def _normalize_text(self, s: str) -> str:
        return normalize_text(s)

    def _tokenize(self, s: str) -> List[str]:
        return tokenize(s)

    def _variants(self, token: str) -> set:
        return variants(token)

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

        # --- Coverage beats fusion ---------------------------------------
        # RRF ADDS ranks, so a document that scores decently in BOTH legs beats
        # one that wins a single leg outright. That is usually right and here it
        # was badly wrong: for 'ARDELL PESTAÑAS REMY LASH 775' the exact article
        # (63984) topped BM25 at 28.7 while the embedder — which has no idea
        # what 775 means — never surfaced it, so three NAKED LASH products,
        # merely decent in both legs, out-summed it and the exact match never
        # reached the candidate list at all. Same story for 'KATAI GELFIX ROMA',
        # where every Gelfix shares the brand and only the city discriminates.
        #
        # So documents that account for EVERY known token of the query go first,
        # keeping fused order among themselves. Covering everything the client
        # said is stronger evidence than a good average across two rankings, and
        # it is the same coverage rule the history layer and the match guard
        # already use. When nothing covers everything, fused order stands.
        known_q = [t for t in dict.fromkeys(self._tokenize(query)) if self._known(t)]
        if known_q:
            # ONE rule (coverage_rank): full coverage first, tightest name next.
            # RRF adds ranks, so a document decent in both legs beats one that
            # wins a single leg outright — which buried exact lexical matches
            # the embedder could not understand ("775", "ROMA").
            ranked = coverage_rank(" ".join(known_q), [
                (pid, (d.get("content") or ""), (d.get("DescripcionArticulo") or ""))
                for pid in fused_ids if (d := self.documents_map.get(pid))])
            full = [k for k, cov, _ in ranked if cov == len(known_q)]
            if full and len(full) < len(fused_ids):
                seen = set(full)
                fused_ids = full + [i for i in fused_ids if i not in seen]
                # Article codes, not point ids: a line of UUIDs cannot be
                # cross-checked against the catalog, which is the only reason
                # this line exists.
                logger.info(f"[RAG] q={query!r}: {len(full)} cubren TODO; ceñido="
                            + ", ".join(
                                f"{(self.documents_map.get(k) or {}).get('CodigoArticulo') or k}:+{x}"
                                for k, c, x in ranked[:6] if c == len(known_q)))

        # --- Default: RRF order with per-code diversity ---
        final_docs = self._pick_topk_with_diversity(fused_ids, top_k)
        # The blind spot: dense and sparse are logged, and so is what the match
        # helper answers, but never WHAT WAS ON THE TABLE in between. An exact
        # sparse hit that never reaches the candidate list looks exactly like a
        # model that ignored it, and the two need opposite fixes.
        logger.info(
            f"[RAG] q={query!r} fusión top5={[i for i in fused_ids[:5]]} -> "
            f"candidatos={[str(d.get('CodigoArticulo') or d.get('id')) for d in final_docs]}")
        score_map = dict(fused)
        q_tokens = list(dict.fromkeys(self._tokenize(query)))
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


# =============================================================================
# CLIENT ORDER HISTORY — resolving informal wording against what THIS client
# has actually bought.
# =============================================================================
"""
A recurring client never says "KG001399" or "ARDELL DELINEADOR PESTAÑAS NEGRO
10GR". They say "el shampoo de coco". Against 8000 articles that is
unresolvable; against the 40-300 that this particular client has ever ordered,
it usually resolves DETERMINISTICALLY — no embeddings, no LLM.

So this layer touches neither Qdrant nor cima: it is token coverage over a
small set. It lives in this module, and not in one of its own, because it must
share the normalizer and the ES<->EN lexicon above. If the history normalized
"champú" differently from the index, the two would silently disagree.

SCOPE — this is a RESOLUTION mechanism, not a feature
-----------------------------------------------------
Reading past orders tells the bot WHICH product a client means. It does not
give it the ability to discuss past orders; the bot's only job is to create
orders. The order statistics (`veces`, `ultima_fecha`, `qty_moda`) exist to
rank candidates and to decide resolved vs ambiguous, and they stay in code.
Only HistoryMatch.agent_view() — code, name, brand — may reach the prompt.
See the guard on that method.

Why the Postgres cache
----------------------
The SQL Server link is read-only, sits behind a VPN that drops idle
connections, and uses NullPool (a fresh ODBC connection per query; see
database.py). Querying it every turn is slow and fragile. Instead:

  cache absent   -> sync inline, with a bounded timeout
  cache fresh    -> use as-is
  cache stale    -> use as-is AND refresh in the background
  SAGE down      -> use the stale cache; if there is none, no history

None of this raises: with no history the bot falls back to the catalog, which
is exactly today's behaviour.

Data quirks, verified against real dumps
----------------------------------------
* SELECT TOP 1000 without ORDER BY over LineasPedidoCliente returns junk rows:
  CodigoEmpresa=0, NumeroPedido=0, CodigoArticulo=''. Filtered out.
* The Cabecera<->Lineas join is FOUR columns (CodigoEmpresa, EjercicioPedido,
  SeriePedido, NumeroPedido) and SeriePedido arrives as an empty string, not
  NULL, in the real data. Were it NULL anywhere, strict equality would drop
  that order silently — hence the "tolerant" COALESCE retry, which is reported
  in the stats so it is visible if it ever fires.
* Lines carry DescripcionArticulo FROZEN at order time, which may differ from
  the current catalog. Free extra names for matching.
* CodigodelCliente is the client's own reference for the article. Where
  populated it is pure gold; the stats count how many lines carry it.
* K_BOT = 0 is the ingestion's catalog filter (see ai/qdrant.py). A history
  product with K_BOT != 0 is not bot-sellable: it is stored flagged and
  excluded from matching, but COUNTED, because if a client's most-ordered
  product is excluded that is a configuration problem worth seeing.
"""

# Company whose orders are read. Real data lives in 1; CodigoEmpresa=0 is residue.
SAGE_EMPRESA = int(os.getenv("SAGE_CODIGO_EMPRESA", "1"))
# Time window: what a client ordered 8 years ago says nothing about today, and
# this bounds the query cost.
# TEMPORARY, and reversible with one variable. Resolve ONLY against what THIS
# client has already bought, never against the general catalog.
#
# Why: of the 8003 articles the catalog exposes, 6762 have not been ordered by
# anyone since 2025, while 3326 that clients DO buy this year are excluded from
# it — plus testers, empty display units and duplicated references. A client's
# own order history has none of that: everything in it is something they really
# bought, under their own CodigoCliente.
#
# Lives here because handlers and llm both need it and rag is below both, so no
# import cycle. Set RAG_HISTORY_ONLY=false to restore catalog fallback; every
# code path for it is intact, just skipped.
HISTORY_ONLY = os.getenv("RAG_HISTORY_ONLY", "true").strip().lower() not in ("0", "false", "no")

HISTORY_YEARS = float(os.getenv("CLIENT_HISTORY_YEARS", "5"))
# Raw order lines pulled per client. This is a SAFETY VALVE, not a tuning knob:
# hitting it means the client's history is INCOMPLETE and the layer will resolve
# against a random-ish subset. At 4000 a wholesaler with 260 orders got 147 of
# them and 35 of its 144 sellable articles — so a product it buys regularly was
# invisible, and the bot offered a TESTER from the excluded pile instead.
# Truncation is logged loudly below; if you see that warning, raise this rather
# than living with a half-read history.
HISTORY_MAX_LINES = int(os.getenv("CLIENT_HISTORY_MAX_LINES", "25000"))
HISTORY_TTL_HOURS = float(os.getenv("CLIENT_HISTORY_TTL_HOURS", "24"))
HISTORY_SYNC_TIMEOUT_S = float(os.getenv("CLIENT_HISTORY_SYNC_TIMEOUT_S", "25"))
# SQL Server caps a statement at 2100 parameters; IN() lists go in chunks.
_IN_CHUNK = 400
# Only K_BOT = 0 is bot-sellable (same criterion as the ingestion).
BOT_VISIBLE_K = K_BOT_VENDIBLE   # -1 (corregido; antes 0, invertido)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if isinstance(v, Decimal):
            return int(v)
        return int(float(v))
    except Exception:
        return default


@dataclass
class SyncStats:
    """What happened during a sync. The primary diagnostic output."""
    codigo_cliente: str = ""
    status: str = "ok"                 # ok | empty | error | cached
    n_lines: int = 0                   # order lines read
    n_orders: int = 0                  # distinct orders
    n_products: int = 0                # distinct sellable articles
    n_excluded: int = 0                # articles with K_BOT != 0
    n_unknown: int = 0                 # history codes absent from Articulos
    n_ref_cliente: int = 0             # lines carrying CodigodelCliente
    tolerant_join: bool = False        # the COALESCE retry was needed
    elapsed_ms: float = 0.0
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ProductRow:
    """One article from a client's history, ready to be matched."""
    codigo: str
    nombre: str = ""                            # current catalog description
    nombres_historicos: List[str] = field(default_factory=list)
    marca: str = ""
    ref_cliente: str = ""
    veces: int = 0                              # distinct orders containing it
    unidades_total: int = 0
    qty_moda: int = 1                           # this client's usual quantity
    primera_fecha: Optional[datetime] = None
    ultima_fecha: Optional[datetime] = None
    bot_visible: bool = True

    def searchable_text(self) -> str:
        """Every piece of text by which this article can be named.

        The CODE belongs here. When two references share a name the bot asks
        the client to choose by code — and then could not find what it had just
        offered, answering "no encuentro ese código" to a code it had printed
        two lines earlier. A client naming a code is naming the product.
        """
        parts = [self.codigo, self.nombre, *self.nombres_historicos,
                 self.marca, self.ref_cliente]
        return " ".join(p for p in parts if p)


@dataclass
class HistoryMatch:
    codigo: str
    nombre: str
    marca: str
    veces: int
    qty_moda: int
    ultima_fecha: Optional[datetime]
    covered: List[str] = field(default_factory=list)     # client tokens covered
    uncovered: List[str] = field(default_factory=list)   # tokens NOBODY covers
    head_match: bool = False                             # the head noun matches
    full_coverage: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "codigo": self.codigo, "nombre": self.nombre, "marca": self.marca,
            "veces": self.veces, "qty_moda": self.qty_moda,
            "ultima_fecha": self.ultima_fecha.isoformat() if self.ultima_fecha else None,
            "cubre": self.covered, "no_cubre": self.uncovered,
            "sustantivo_ok": self.head_match, "cobertura_total": self.full_coverage,
        }

    def agent_view(self) -> Dict[str, str]:
        """The ONLY shape of a history match that may reach the LLM.

        SCOPE GUARD — read before widening this.

        The order history is a RESOLUTION mechanism: it turns "el shampoo de
        coco" into a catalog code. It does NOT give the bot the ability to talk
        about past orders. The bot's only job is to create orders.

        So `veces`, `ultima_fecha`, `qty_moda` and the coverage internals stay
        in code, where they are used for ranking and for the resolved/ambiguous
        verdict. They must never be rendered into the prompt, because a model
        that can see them WILL narrate them ("como la última vez, 6 unidades")
        and will then happily field "¿qué pedí el mes pasado?" — which is
        outside its remit and turns a lookup into a data-exposure path.

        as_dict() keeps everything: that one is for the operator's diagnostics,
        which never touch the model.
        """
        return {"codigo": self.codigo, "nombre": self.nombre, "marca": self.marca}


@dataclass
class HistoryResolution:
    """Verdict for ONE client mention."""
    query: str
    verdict: str = "none"       # resolved | ambiguous | partial | blocked | none | disabled
    tokens: List[str] = field(default_factory=list)
    matches: List[HistoryMatch] = field(default_factory=list)
    # Full matches this client BUYS but K_BOT excludes from the bot's catalog.
    # Surfaced rather than dropped: they are the reason a client can reject
    # every option offered, and substituting past them is the worst outcome.
    blocked: List[HistoryMatch] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"consulta": self.query, "veredicto": self.verdict,
                "tokens": self.tokens, "motivo": self.reason,
                "candidatos": [m.as_dict() for m in self.matches],
                "no_vendibles": [m.as_dict() for m in self.blocked]}


class ClientHistoryRetriever:
    """SAGE order history per client: read, cache, match."""

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        self._bg_tasks: set = set()

    # ------------------------------------------------------------- internals
    def _lock_for(self, code: str) -> asyncio.Lock:
        lock = self._locks.get(code)
        if lock is None:
            lock = self._locks[code] = asyncio.Lock()
        if len(self._locks) > 2000:
            for k in [k for k, v in self._locks.items() if not v.locked()][:1000]:
                self._locks.pop(k, None)
        return lock

    @staticmethod
    def _line_select(tolerant: bool):
        """Order lines + header date. Only mapped columns are projected, so the
        query does not drag the ~180 junk columns of LineasPedidoCliente."""
        keys = [
            MSOrderLine.CodigoEmpresa == MSOrderHeader.CodigoEmpresa,
            MSOrderLine.EjercicioPedido == MSOrderHeader.EjercicioPedido,
            MSOrderLine.NumeroPedido == MSOrderHeader.NumeroPedido,
        ]
        if tolerant:
            keys.append(func.coalesce(MSOrderLine.SeriePedido, "") ==
                        func.coalesce(MSOrderHeader.SeriePedido, ""))
        else:
            keys.append(MSOrderLine.SeriePedido == MSOrderHeader.SeriePedido)
        return (
            select(
                MSOrderHeader.EjercicioPedido,
                MSOrderHeader.SeriePedido,
                MSOrderHeader.NumeroPedido,
                MSOrderHeader.FechaPedido,
                MSOrderLine.CodigoArticulo,
                MSOrderLine.DescripcionArticulo,
                MSOrderLine.Descripcion2Articulo,
                MSOrderLine.CodigodelCliente,
                MSOrderLine.UnidadesPedidas,
            )
            # Explicit FROM: otherwise SQLAlchemy infers it from the first
            # projected column, which is brittle if the SELECT is reordered.
            .select_from(MSOrderHeader)
            .join(MSOrderLine, and_(*keys))
        )

    async def _fetch_lines(self, code: str, tolerant: bool) -> List[Any]:
        cutoff = datetime.now() - timedelta(days=int(HISTORY_YEARS * 365.25))
        stmt = (
            self._line_select(tolerant)
            .where(
                MSOrderHeader.CodigoCliente == code,
                MSOrderHeader.CodigoEmpresa == SAGE_EMPRESA,
                MSOrderHeader.FechaPedido >= cutoff,
                MSOrderLine.CodigoArticulo.isnot(None),
                MSOrderLine.CodigoArticulo != "",
            )
            .order_by(MSOrderHeader.FechaPedido.desc())
            .limit(HISTORY_MAX_LINES)
        )
        async with AsyncSessionSQL() as s:
            return list((await s.execute(stmt)).all())

    @staticmethod
    async def _fetch_catalog(codes: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Current description, brand and K_BOT for the given codes."""
        out: Dict[str, Dict[str, Any]] = {}
        codes = [c for c in dict.fromkeys(codes) if c]
        if not codes:
            return out
        async with AsyncSessionSQL() as s:
            for i in range(0, len(codes), _IN_CHUNK):
                chunk = codes[i:i + _IN_CHUNK]
                stmt = select(
                    MSArticle.CodigoArticulo,
                    MSArticle.DescripcionArticulo,
                    MSArticle.Descripcion2Articulo,
                    MSArticle.MarcaProducto,
                    MSArticle.K_BOT,
                    MSArticle.CodigoAreaCompetenciaLc,
                    MSArticle.CodigoAlternativo,
                    MSArticle.FechaAlta,
                ).where(
                    MSArticle.CodigoArticulo.in_(chunk)
                # The same article repeats once per CodigoEmpresa. The ingestion
                # dedupes by code; we do the same, but with an explicit ORDER BY
                # so the SAME row always wins (the ingestion has this very
                # nondeterminism, documented in qdrant.py's _to_point_id).
                ).order_by(MSArticle.CodigoArticulo, MSArticle.CodigoEmpresa)
                for row in (await s.execute(stmt)).all():
                    key = (row.CodigoArticulo or "").strip()
                    if key and key not in out:
                        out[key] = {
                            "nombre": (row.DescripcionArticulo or "").strip(),
                            "nombre2": (row.Descripcion2Articulo or "").strip(),
                            "marca": (row.MarcaProducto or "").strip(),
                            "k_bot": _as_int(row.K_BOT, -1),
                            # pertenencia al corpus filtrado: mismo es_vendible
                            "vendible": es_vendible(row.CodigoAreaCompetenciaLc,
                                                    row.DescripcionArticulo, row.K_BOT),
                            "ean": (row.CodigoAlternativo or "").strip(),
                            "fecha_alta": row.FechaAlta,
                        }
        return out

    @staticmethod
    def _aggregate(rows: Iterable[Any], catalog: Dict[str, Dict[str, Any]]
                   ) -> Tuple[List[ProductRow], SyncStats]:
        """Lines -> one ProductRow per article.

        Done in Python on purpose: the mode of the quantities and the set of
        historical names are awkward in T-SQL, and the per-client volume is
        small enough that it does not matter.
        """
        stats = SyncStats()
        orders: set = set()
        per_code: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"orders": set(), "units": 0, "qtys": Counter(), "names": set(),
                     "ref": "", "first": None, "last": None})

        for r in rows:
            code = (r.CodigoArticulo or "").strip()
            if not code:
                continue
            stats.n_lines += 1
            order_key = (r.EjercicioPedido, (r.SeriePedido or ""), r.NumeroPedido)
            orders.add(order_key)
            bucket = per_code[code]
            bucket["orders"].add(order_key)
            qty = _as_int(r.UnidadesPedidas, 0)
            if qty > 0:
                bucket["units"] += qty
                bucket["qtys"][qty] += 1
            for name in ((r.DescripcionArticulo or "").strip(),
                         (r.Descripcion2Articulo or "").strip()):
                if name:
                    bucket["names"].add(name)
            ref = (r.CodigodelCliente or "").strip()
            if ref:
                stats.n_ref_cliente += 1
                bucket["ref"] = ref
            fecha = r.FechaPedido
            if fecha:
                bucket["first"] = fecha if bucket["first"] is None else min(bucket["first"], fecha)
                bucket["last"] = fecha if bucket["last"] is None else max(bucket["last"], fecha)

        stats.n_orders = len(orders)
        products: List[ProductRow] = []
        for code, b in per_code.items():
            meta = catalog.get(code)
            if meta is None:
                stats.n_unknown += 1
            k_bot = meta["k_bot"] if meta else -1
            # Pertenencia al corpus filtrado (mismo es_vendible que la ingesta):
            # no basta K_BOT, también MERCH y palabras. Un producto que el
            # cliente compró pero ya no es vendible queda NO visible -> va a la
            # pila `blocked` (salvaguarda caso 11: no se ofrece, pero se conoce
            # para no sustituirlo en silencio).
            visible = bool(meta["vendible"]) if meta else False
            if visible:
                stats.n_products += 1
            else:
                stats.n_excluded += 1
            names = sorted(b["names"])
            current = (meta or {}).get("nombre") or (names[0] if names else "")
            name2 = (meta or {}).get("nombre2") or ""
            historic = [n for n in names if n != current]
            if name2:
                historic.append(name2)
            products.append(ProductRow(
                codigo=code,
                nombre=current,
                nombres_historicos=historic,
                marca=(meta or {}).get("marca", ""),
                ref_cliente=b["ref"],
                veces=len(b["orders"]),
                unidades_total=b["units"],
                qty_moda=(b["qtys"].most_common(1)[0][0] if b["qtys"] else 1),
                primera_fecha=b["first"],
                ultima_fecha=b["last"],
                bot_visible=visible,
            ))
        products.sort(key=lambda p: (-p.veces, p.codigo))
        return products, stats

    # ------------------------------------------------------- Postgres cache
    @staticmethod
    async def _write_cache(code: str, products: List[ProductRow], stats: SyncStats) -> None:
        """Full replacement of that client's history. Simple and correct: the
        volume is small, which avoids reasoning about partial diffs."""
        async with AsyncSessionPG() as db:
            await db.execute(delete(ClientProductHistory)
                             .where(ClientProductHistory.codigo_cliente == code))
            for p in products:
                db.add(ClientProductHistory(
                    codigo_cliente=code, codigo_articulo=p.codigo,
                    descripcion=p.nombre,
                    descripciones_historicas=p.nombres_historicos,
                    marca=p.marca, ref_cliente=p.ref_cliente,
                    bot_visible=p.bot_visible, veces=p.veces,
                    unidades_total=p.unidades_total, qty_moda=p.qty_moda,
                    primera_fecha=p.primera_fecha, ultima_fecha=p.ultima_fecha,
                    refreshed_at=_now_utc(),
                ))
            sync = (await db.execute(select(ClientHistorySync)
                                     .where(ClientHistorySync.codigo_cliente == code))
                    ).scalars().first()
            if sync is None:
                sync = ClientHistorySync(codigo_cliente=code)
                db.add(sync)
            sync.refreshed_at = _now_utc()
            sync.attempted_at = _now_utc()
            sync.status = stats.status
            sync.n_products = stats.n_products
            sync.n_lines = stats.n_lines
            sync.n_orders = stats.n_orders
            sync.n_excluded = stats.n_excluded
            sync.n_ref_cliente = stats.n_ref_cliente
            sync.tolerant_join = stats.tolerant_join
            sync.error = (stats.error or "")[:500]
            await db.commit()

    @staticmethod
    async def _mark_error(code: str, err: str) -> None:
        async with AsyncSessionPG() as db:
            sync = (await db.execute(select(ClientHistorySync)
                                     .where(ClientHistorySync.codigo_cliente == code))
                    ).scalars().first()
            if sync is None:
                sync = ClientHistorySync(codigo_cliente=code)
                db.add(sync)
            sync.status = "error"
            sync.error = (err or "")[:500]
            sync.attempted_at = _now_utc()
            await db.commit()

    @staticmethod
    async def _read_cache(code: str) -> Tuple[List[ProductRow], Optional[Any]]:
        async with AsyncSessionPG() as db:
            sync = (await db.execute(select(ClientHistorySync)
                                     .where(ClientHistorySync.codigo_cliente == code))
                    ).scalars().first()
            rows = list((await db.execute(
                select(ClientProductHistory)
                .where(ClientProductHistory.codigo_cliente == code)
                .order_by(ClientProductHistory.veces.desc())
            )).scalars().all())
        products = [ProductRow(
            codigo=r.codigo_articulo, nombre=r.descripcion or "",
            nombres_historicos=list(r.descripciones_historicas or []),
            marca=r.marca or "", ref_cliente=r.ref_cliente or "",
            veces=r.veces or 0, unidades_total=_as_int(r.unidades_total, 0),
            qty_moda=r.qty_moda or 1, primera_fecha=r.primera_fecha,
            ultima_fecha=r.ultima_fecha, bot_visible=bool(r.bot_visible),
        ) for r in rows]
        return products, sync

    @staticmethod
    def _is_stale(sync: Optional[Any]) -> bool:
        if sync is None or sync.refreshed_at is None:
            return True
        ref = sync.refreshed_at
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        return (_now_utc() - ref) > timedelta(hours=HISTORY_TTL_HOURS)

    # ------------------------------------------------------------- sync API
    def _sim_products(self, code: str) -> Optional[Tuple[List["ProductRow"], "SyncStats"]]:
        """Fake history for a test client, or None when this is a real client.

        None vs an empty list matters: None means "not simulated, go to SAGE",
        while an empty list is a deliberately empty simulated history (the
        "client with no purchases" test case).
        """
        rows = sim_history_cache.get(code)
        if rows is None:
            return None
        products = [ProductRow(
            codigo=str(r.get("codigo") or ""),
            nombre=str(r.get("nombre") or ""),
            nombres_historicos=list(r.get("nombres_historicos") or []),
            marca=str(r.get("marca") or ""),
            ref_cliente=str(r.get("ref_cliente") or ""),
            veces=_as_int(r.get("veces"), 1),
            unidades_total=_as_int(r.get("unidades_total"), 0),
            qty_moda=_as_int(r.get("qty_moda"), 1),
            primera_fecha=r.get("primera_fecha"),
            ultima_fecha=r.get("ultima_fecha"),
            bot_visible=bool(r.get("bot_visible", True)),
        ) for r in rows if r.get("codigo")]
        stats = SyncStats(
            codigo_cliente=code, status="sim",
            n_products=len([p for p in products if p.bot_visible]),
            n_orders=max((p.veces for p in products), default=0),
            n_lines=sum(p.veces for p in products),
            n_ref_cliente=sum(1 for p in products if p.ref_cliente),
        )
        return products, stats

    async def refresh(self, code: str) -> SyncStats:
        """Read SAGE and rewrite the cache. Never raises: returns stats.error."""
        code = (code or "").strip()
        if not code:
            return SyncStats(status="error", error="codigo_cliente vacío")
        # A simulated client has no SAGE rows and must never be written to
        # Postgres; a forced refresh on one is a no-op that reports the truth.
        sim = self._sim_products(code)
        if sim is not None:
            return sim[1]
        t0 = asyncio.get_running_loop().time()
        async with self._lock_for(code):
            try:
                rows = await self._fetch_lines(code, tolerant=False)
                tolerant = False
                if not rows:
                    # Zero lines because SeriePedido is NULL, or because the
                    # client has no orders? Telling those apart matters: the
                    # first is a data bug, the second is normal.
                    rows = await self._fetch_lines(code, tolerant=True)
                    tolerant = bool(rows)
                    if tolerant:
                        hist_logger.warning(
                            f"[HIST] {code}: strict join returned 0 lines and the "
                            f"tolerant one {len(rows)} — SeriePedido is inconsistent.")
                catalog = await self._fetch_catalog(
                    [(r.CodigoArticulo or "").strip() for r in rows])
                products, stats = self._aggregate(rows, catalog)
                stats.codigo_cliente = code
                stats.tolerant_join = tolerant
                stats.status = "ok" if products else "empty"
                stats.elapsed_ms = (asyncio.get_running_loop().time() - t0) * 1000
                await self._write_cache(code, products, stats)
                hist_logger.info(
                    f"[HIST] {code}: {stats.n_products} sellable articles from "
                    f"{stats.n_orders} orders ({stats.n_lines} lines, "
                    f"{stats.n_excluded} excluded by K_BOT, {stats.n_unknown} "
                    f"without a catalog row) in {stats.elapsed_ms:.0f}ms")
                # A truncated history is WORSE than no history: it resolves
                # confidently against a subset, so a product the client buys
                # every month can be missing while a tester from the excluded
                # pile gets offered in its place. That happened, and the only
                # sign in the log was a suspiciously round line count.
                if len(rows) >= HISTORY_MAX_LINES:
                    hist_logger.warning(
                        f"[HIST] {code}: TRUNCADO en {HISTORY_MAX_LINES} líneas "
                        f"(CLIENT_HISTORY_MAX_LINES). Sólo se han leído los "
                        f"{stats.n_orders} pedidos más recientes, así que el historial "
                        f"está INCOMPLETO y puede resolver mal. Sube el límite.")
                return stats
            except Exception as e:
                hist_logger.error(f"[HIST] Failed reading history for {code}: {e}")
                await self._mark_error(code, str(e))
                return SyncStats(codigo_cliente=code, status="error", error=str(e),
                                 elapsed_ms=(asyncio.get_running_loop().time() - t0) * 1000)

    def _refresh_in_background(self, code: str) -> None:
        task = asyncio.create_task(self.refresh(code))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    @staticmethod
    def _looks_capped(n_lines: int) -> bool:
        """Was this cached history cut off by a line cap rather than complete?

        A cap is always a round number someone typed: 4000, 10000, 25000. A real
        client's line total landing on an exact multiple of 1000 happens about
        once in a thousand — and the cost of being wrong is one extra sync,
        against the cost of being right being a history that silently answers
        with a third of the client's catalog.

        Only counts BELOW the current cap matter: at or above it, the history is
        either genuinely that big or capped right now, and re-reading changes
        nothing.
        """
        return 0 < n_lines < HISTORY_MAX_LINES and n_lines % 1000 == 0

    async def load(self, code: str, *, force: bool = False
                   ) -> Tuple[List[ProductRow], SyncStats]:
        """This client's products, from cache, syncing when necessary."""
        code = (code or "").strip()
        if not code:
            return [], SyncStats(status="error", error="codigo_cliente vacío")

        # 1) SIMULATED history (test clients). Checked FIRST, mirroring the
        #    lookup order client resolution already uses, and it never touches
        #    a database: a simulation must not leave rows behind. `force` is
        #    meaningless here — there is no source to re-read.
        sim = self._sim_products(code)
        if sim is not None:
            return sim

        if force:
            stats = await self.refresh(code)
            products, _ = await self._read_cache(code)
            return products, stats

        products, sync = await self._read_cache(code)
        # A cache written while the line cap was LOWER is a truncated history
        # wearing a fresh timestamp: it resolves confidently against a subset
        # and the TTL keeps it alive for a day. Raising the cap must therefore
        # invalidate it, or the fix never takes effect — which is exactly what
        # happened when 4000 became 25000 and the layer went on serving 35 of
        # the client's 144 articles, a tester among them.
        if sync is not None and self._looks_capped(sync.n_lines or 0):
            hist_logger.warning(
                f"[HIST] {code}: la caché tiene exactamente {sync.n_lines} líneas — "
                f"eso es un tope anterior, no un total real (ahora "
                f"{HISTORY_MAX_LINES}). Se resincroniza.")
            sync = None
        if sync is None or sync.refreshed_at is None:
            # First time: we have to wait, but with a timeout — a slow SAGE
            # cannot block the turn indefinitely.
            try:
                stats = await asyncio.wait_for(self.refresh(code),
                                               timeout=HISTORY_SYNC_TIMEOUT_S)
            except asyncio.TimeoutError:
                hist_logger.warning(f"[HIST] Timeout ({HISTORY_SYNC_TIMEOUT_S}s) syncing "
                                    f"{code}; no history this turn.")
                return [], SyncStats(codigo_cliente=code, status="error", error="timeout")
            products, _ = await self._read_cache(code)
            return products, stats

        stats = SyncStats(codigo_cliente=code, status="cached",
                          n_products=len([p for p in products if p.bot_visible]),
                          n_lines=sync.n_lines or 0, n_orders=sync.n_orders or 0,
                          n_excluded=sync.n_excluded or 0,
                          n_ref_cliente=sync.n_ref_cliente or 0,
                          tolerant_join=bool(sync.tolerant_join))
        if self._is_stale(sync):
            hist_logger.info(f"[HIST] Cache for {code} is stale; refreshing in background.")
            self._refresh_in_background(code)
        return products, stats

    # ------------------------------------------------------------- matching
    @staticmethod
    def _covers(token: str, doc_tokens: set) -> bool:
        """The client's token appears in the product, across languages
        ('shampoo' covers 'CHAMPU') via the lexicon at the top of this module."""
        return any(v in doc_tokens for v in variants(token))

    def match(self, query: str, products: Sequence[ProductRow],
              limit: int = 5) -> HistoryResolution:
        """Match one informal mention against a client's history.

        Token coverage, not similarity. The denominator is EVERY content token
        the client used: if they say 'nivea' and no product in their history
        carries 'nivea', that is NOT ignored — it is the signal that the
        product is not in their history. Same honesty as `query_uncovered` in
        the hybrid path: we would rather admit we do not have it than silently
        substitute something else.

        NOT-SELLABLE MATCHES. Articles the client has bought but that K_BOT
        excludes are matched too, and reported separately as `blocked`. They
        used to be invisible, and that produced the worst possible outcome:
        the client asked for an adhesive they buy regularly, the history
        silently had nothing to say, and the catalog fallback committed a
        DIFFERENT adhesive. Knowing the client means a product we may not sell
        is far more useful than pretending we never heard of it.
        """
        res = HistoryResolution(query=query)
        q_tokens = list(dict.fromkeys(tokenize(query)))
        res.tokens = q_tokens
        if not q_tokens:
            res.reason = "the mention leaves no content tokens"
            return res

        if not products:
            res.reason = "this client has no history"
            return res

        head = q_tokens[0]   # Spanish is noun-first: "_champú_ de coco"
        scored: List[HistoryMatch] = []
        blocked: List[HistoryMatch] = []
        for p in products:
            doc_tokens = set(tokenize(p.searchable_text()))
            covered = [t for t in q_tokens if self._covers(t, doc_tokens)]
            if not covered:
                continue
            m = HistoryMatch(
                codigo=p.codigo, nombre=p.nombre, marca=p.marca, veces=p.veces,
                qty_moda=p.qty_moda, ultima_fecha=p.ultima_fecha,
                covered=covered,
                uncovered=[t for t in q_tokens if t not in covered],
                head_match=self._covers(head, doc_tokens),
                full_coverage=len(covered) == len(q_tokens),
            )
            (scored if p.bot_visible else blocked).append(m)

        res.blocked = [m for m in blocked if m.full_coverage][:limit]

        if not scored:
            res.reason = ("this client buys a full match the bot may not sell "
                          f"({', '.join(m.codigo for m in res.blocked)})"
                          if res.blocked else
                          "no article in the history shares any term")
            if res.blocked:
                res.verdict = "blocked"
            return res

        sort_key = lambda m: (
            -len(m.covered),                                         # 1) most terms covered
            -(1 if m.head_match else 0),                             # 2) head noun matches
            -m.veces,                                                # 3) ordered more often
            -(m.ultima_fecha.timestamp() if m.ultima_fecha else 0),   # 4) more recent
        )
        scored.sort(key=sort_key)

        full = [m for m in scored if m.full_coverage]
        if len(full) == 1:
            res.verdict, res.matches = "resolved", full[:1]
            res.reason = "exactly one history article covers everything they said"
        elif len(full) > 1:
            res.verdict, res.matches = "ambiguous", full[:limit]
            res.reason = f"{len(full)} history articles cover everything they said"
        elif res.blocked:
            # PRECEDENCE. A blocked FULL match beats any partial visible one.
            # This ordering is the whole point: the client has other sellable
            # articles of the same brand, so `scored` is never empty and the
            # earlier version fell straight to "partial" -> catalog, leaving the
            # blocked article invisible. That is exactly how "adhesivo ardell
            # transparente" ended up committing a different adhesive: the
            # article the client actually buys was excluded by K_BOT, and three
            # other Ardell products covering only the word "ardell" were enough
            # to mask it.
            res.verdict = "blocked"
            res.reason = ("the client buys a full match the bot may not sell: "
                          + ", ".join(m.codigo for m in res.blocked))
        else:
            # Partial coverage is NEVER treated as resolved. It is returned so
            # the caller can decide: ask, or fall through to the full catalog.
            res.verdict, res.matches = "partial", scored[:limit]
            res.reason = "partial matches only"
        if res.blocked and res.verdict in ("resolved", "ambiguous"):
            res.reason += (f" (also buys {len(res.blocked)} matching article(s) "
                           f"the bot may not sell)")
        return res

    async def resolve_queries(self, code: str, queries: Sequence[str], *,
                              force: bool = False
                              ) -> Tuple[List[HistoryResolution], SyncStats]:
        """Resolve several mentions against one client's history."""
        products, stats = await self.load(code, force=force)
        return [self.match(q, products) for q in queries], stats


client_history = ClientHistoryRetriever()