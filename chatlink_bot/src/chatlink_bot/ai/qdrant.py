# src/chatlink_bot/ai/qdrant.py
"""
Qdrant service: collection lifecycle, product ingestion, search.

INGESTION — incremental by content hash. The old version re-embedded the
whole catalog (~8000 full LLM prefills, ~15 min) on every run even when
nothing had changed, because it had no memory of what was already indexed.
Now each point's payload carries a `content_hash`; a sync run is:

  1. ONE SQL query loads the source catalog (K_BOT = 0), deduped by code
     (the old OFFSET/LIMIT pagination issued ~125 queries).
  2. ONE cheap Qdrant scroll reads existing {id: content_hash}.
  3. Diff: embed & upsert only NEW or CHANGED products; DELETE stale points
     (removed from SAGE or K_BOT flipped) so the index mirrors the source.
  4. Changed rows are embedded in batches with bounded concurrency
     (INGEST_EMBED_CONCURRENCY) and upserted as each batch completes, so
     Qdrant writes overlap GPU embedding instead of serializing behind it.

Result: the daily/startup run on an unchanged catalog costs a few seconds
(SQL + scroll + hashes); GPU time is paid only for actual changes. The very
first ingest is still embedding-bound — lower QDRANT_PRODUCT_TEXT_MAX_CHARS
if that one-off matters.
"""
import asyncio
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import AsyncQdrantClient, models as qmodels
from sqlalchemy import select

from .cima_client import CimaClient
from ..database import AsyncSessionSQL
from ..models import MSArticle

logger = logging.getLogger("Qdrant")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_FALLBACK_URL = os.getenv("QDRANT_FALLBACK_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "products")
QDRANT_RESET_ON_STARTUP = os.getenv("QDRANT_RESET_ON_STARTUP", "false").lower() == "true"

CIMA_URL = os.getenv("CIMA_URL", "http://cima:8000")
EMBED_MODEL = os.getenv("CIMA_MODEL", "unsloth/gemma-4-E2B-it-GGUF:Q8_0")

EMBED_BATCH = int(os.getenv("INGEST_EMBED_BATCH", "64"))
# In-flight embed requests. DEFAULT 1: two parallel 64-text embeds crashed the
# CUDA context on a 6 GiB card (CUresult 999) in production. Raise only if
# cima is configured for parallel serving AND has VRAM headroom.
EMBED_CONCURRENCY = max(1, int(os.getenv("INGEST_EMBED_CONCURRENCY", "1")))
# Per-batch retries for transient 5xx, with linear backoff.
BATCH_RETRIES = max(1, int(os.getenv("INGEST_BATCH_RETRIES", "3")))
RETRY_BASE_S = float(os.getenv("INGEST_RETRY_BASE_S", "5"))
# Circuit breaker: this many CONSECUTIVE failed batches = the engine is down
# (e.g. dead CUDA context) -> abort the run instead of erroring 120 more times.
ABORT_AFTER_FAILURES = max(1, int(os.getenv("INGEST_ABORT_AFTER_FAILURES", "3")))


@dataclass(frozen=True)
class _SourceDoc:
    """One catalog product prepared for indexing."""
    point_id: Any                 # int (numeric codes) or deterministic UUID str
    text: str                     # what gets embedded
    payload: Dict[str, Any]       # stored alongside the vector (includes content_hash)


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class QdrantService:
    def __init__(self) -> None:
        self.primary_url = QDRANT_URL
        self.fallback_url = QDRANT_FALLBACK_URL
        self.client = AsyncQdrantClient(url=self.primary_url, prefer_grpc=False)
        self.collection = QDRANT_COLLECTION

        self._collection_ready = False
        self._lock = asyncio.Lock()
        self._using_fallback = False
        self._cima = CimaClient(CIMA_URL)

    # ------------------------------------------------------------ lifecycle
    async def health_check(self) -> Dict[str, Any]:
        """Connectivity + collection presence, without erroring if missing."""
        try:
            col_res = await self.client.get_collections()
            return {
                "status": "ok",
                "connected": True,
                "collection_exists": any(c.name == self.collection for c in col_res.collections),
                "total_collections": len(col_res.collections),
                "url": self.fallback_url if self._using_fallback else self.primary_url,
            }
        except Exception as e:
            return {"status": "error", "error": str(e),
                    "url": self.fallback_url if self._using_fallback else self.primary_url}

    async def _detect_vector_size(self) -> int:
        """Probe the embedder (waits up to ~300s for the model to load)."""
        for attempt in range(30):
            try:
                resp = await asyncio.to_thread(self._cima.embed, EMBED_MODEL, ["vector_size_probe"])
                vec = (resp.embeddings or [[]])[0]
                if vec:
                    return len(vec)
            except Exception as e:
                logger.warning(f"Waiting for embedding model {EMBED_MODEL} ({attempt + 1}/30)... {e}")
            await asyncio.sleep(10)
        raise RuntimeError(f"Could not detect vector size: {EMBED_MODEL} unavailable after 300s.")

    async def _check_and_init_collection(self, client: AsyncQdrantClient) -> None:
        resp = await client.get_collections()
        exists = any(c.name == self.collection for c in resp.collections)
        if QDRANT_RESET_ON_STARTUP and exists:
            logger.warning(f"RESET enabled: deleting Qdrant collection '{self.collection}'...")
            await client.delete_collection(collection_name=self.collection)
            exists = False
        if not exists:
            vector_size = await self._detect_vector_size()
            logger.info(f"Creating Qdrant collection '{self.collection}' (size={vector_size})...")
            await client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_size, distance=qmodels.Distance.COSINE, on_disk=True),
            )

    async def ensure_ready(self) -> bool:
        if self._collection_ready:
            return True
        async with self._lock:
            if self._collection_ready:
                return True
            try:
                if not self._using_fallback:
                    await self._check_and_init_collection(self.client)
                    self._collection_ready = True
                    return True
            except Exception as e:
                logger.warning(f"Primary Qdrant connection failed: {e}")
            try:
                logger.info(f"Attempting fallback to {self.fallback_url}...")
                await self.client.close()
                self.client = AsyncQdrantClient(url=self.fallback_url, prefer_grpc=False)
                await self._check_and_init_collection(self.client)
                self._using_fallback = True
                self._collection_ready = True
                return True
            except Exception as e:
                logger.error(f"Critical: Qdrant unavailable: {e}")
                return False

    async def get_collection_count(self) -> int:
        if not await self.ensure_ready():
            return 0
        try:
            return int((await self.client.count(collection_name=self.collection)).count)
        except Exception as e:
            logger.warning(f"Failed to get collection count: {e}")
            return 0

    # ------------------------------------------------------------ ingestion
    @staticmethod
    def _build_product_text(a: MSArticle) -> str:
        # CodigoArticulo first improves partial vector matches. Capped because
        # every embed is a full LLM prefill: this cap is THE knob for full
        # re-ingest speed (halve it ≈ halve GPU time) at ~zero retrieval loss
        # on short product queries.
        parts = [
            str(a.CodigoArticulo or "").strip(),
            a.DescripcionArticulo or "",
            a.Descripcion2Articulo or "",
            a.DescripcionLinea or "",
            a.ComentarioArticulo or "",
            a.MarcaProducto or "",
        ]
        text = "\n".join(p.strip() for p in parts if p and p.strip())
        return text[: int(os.getenv("QDRANT_PRODUCT_TEXT_MAX_CHARS", "600"))]

    @staticmethod
    def _to_point_id(codigo_articulo: Any) -> int | str:
        """
        Numeric codes -> int id (stable, unchanged scheme). Alphanumeric ->
        UUID5 of the CODE ONLY: the catalog is deduped by code across
        companies, and the old empresa-qualified UUID flapped between runs
        when the same code existed under two companies (nondeterministic row
        order), causing pointless churn. Existing alnum points are migrated
        once by the stale-deletion pass.
        """
        code = str(codigo_articulo or "").strip()
        return int(code) if code.isdigit() else str(uuid.uuid5(uuid.NAMESPACE_DNS, code))

    async def _load_source_products(self) -> Dict[str, _SourceDoc]:
        """ONE query over SAGE (K_BOT = 0), deduped by code -> {str(point_id): doc}."""
        async with AsyncSessionSQL() as s:
            rows = (await s.execute(
                select(MSArticle).where(MSArticle.K_BOT == 0).order_by(MSArticle.CodigoArticulo)
            )).scalars().all()

        docs: Dict[str, _SourceDoc] = {}
        seen_codes: set[str] = set()
        for a in rows:
            code = str(a.CodigoArticulo or "").strip()
            if not code or code in seen_codes:
                continue  # same article repeats once per company -> one point
            seen_codes.add(code)
            content = self._build_product_text(a)
            if not content:
                continue
            pid = self._to_point_id(code)
            docs[str(pid)] = _SourceDoc(
                point_id=pid,
                text=content,
                payload={
                    "CodigoArticulo": code,
                    "DescripcionArticulo": a.DescripcionArticulo,
                    "Descripcion2Articulo": a.Descripcion2Articulo,
                    "DescripcionLinea": a.DescripcionLinea,
                    "ComentarioArticulo": a.ComentarioArticulo,
                    "MarcaProducto": a.MarcaProducto,
                    "K_BOT": int(a.K_BOT or -1),
                    "content": content,
                    "content_hash": _content_hash(content),
                },
            )
        return docs

    async def _existing_hashes(self) -> Dict[str, str]:
        """ONE scroll pass -> {str(point_id): content_hash} for the whole collection."""
        hashes: Dict[str, str] = {}
        next_offset = None
        while True:
            points, next_offset = await self.client.scroll(
                collection_name=self.collection, limit=1024,
                with_payload=["content_hash"], with_vectors=False, offset=next_offset)
            for p in points:
                hashes[str(p.id)] = (p.payload or {}).get("content_hash") or ""
            if next_offset is None:
                return hashes

    @staticmethod
    def plan_sync(source_hashes: Dict[str, str], existing_hashes: Dict[str, str]
                  ) -> Tuple[List[str], List[str]]:
        """Pure diff -> (ids to embed+upsert, ids to delete). Unit-testable."""
        changed = [pid for pid, h in source_hashes.items() if existing_hashes.get(pid) != h]
        stale = [pid for pid in existing_hashes if pid not in source_hashes]
        return changed, stale

    async def ingest_products_from_sqlserver(self, batch_size: int = EMBED_BATCH) -> int:
        """
        Incremental sync SAGE -> Qdrant. Returns the number of points written
        (0 = catalog unchanged; callers can skip the BM25 rebuild).

        Failure semantics (learned in production): stale points are deleted
        ONLY at the end of a fully clean run. Deleting them first once gutted
        the index when the embedder crashed mid-run — old points must keep
        serving searches until their replacements actually exist. A batch
        retries transient errors with backoff; ABORT_AFTER_FAILURES
        consecutive dead batches trips a circuit breaker (a crashed CUDA
        context fails everything instantly — no point sending 120 more).
        """
        if not await self.ensure_ready():
            return 0

        source = await self._load_source_products()
        existing = await self._existing_hashes()
        changed_ids, stale_ids = self.plan_sync(
            {pid: d.payload["content_hash"] for pid, d in source.items()}, existing)

        logger.info(f"[Ingest] source={len(source)} indexed={len(existing)} "
                    f"changed={len(changed_ids)} stale={len(stale_ids)}")

        written = 0
        failed_batches = 0
        if changed_ids:
            docs = [source[pid] for pid in changed_ids]
            batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
            semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)
            state_lock = asyncio.Lock()
            consecutive_failures = 0
            abort = asyncio.Event()

            async def _embed_with_retries(batch: List[_SourceDoc]) -> List[List[float]]:
                last_error: Exception = RuntimeError("no attempt made")
                for attempt in range(BATCH_RETRIES):
                    if abort.is_set():
                        raise last_error
                    try:
                        resp = await asyncio.to_thread(
                            self._cima.embed, model=EMBED_MODEL, input=[d.text for d in batch])
                        return resp.embeddings or []
                    except Exception as e:
                        last_error = e
                        if attempt < BATCH_RETRIES - 1:
                            await asyncio.sleep(RETRY_BASE_S * (attempt + 1))
                raise last_error

            async def _embed_and_upsert(batch: List[_SourceDoc], index: int) -> str:
                nonlocal written, consecutive_failures, failed_batches
                if abort.is_set():
                    return "skipped"
                async with semaphore:
                    if abort.is_set():
                        return "skipped"
                    try:
                        vectors = await _embed_with_retries(batch)
                    except Exception as e:
                        async with state_lock:
                            failed_batches += 1
                            consecutive_failures += 1
                            tripped = consecutive_failures >= ABORT_AFTER_FAILURES
                            if tripped and not abort.is_set():
                                abort.set()
                                logger.error(
                                    f"[Ingest] ABORTING: {consecutive_failures} consecutive batch "
                                    f"failures — embedding engine looks down ({e}). "
                                    f"Existing points are kept; next sync will resume.")
                        if not abort.is_set():
                            logger.error(f"[Ingest] Batch {index + 1}/{len(batches)} failed "
                                         f"after {BATCH_RETRIES} attempts: {e}")
                        return "failed"
                points = [
                    qmodels.PointStruct(id=d.point_id, vector=vec, payload=d.payload)
                    for d, vec in zip(batch, vectors) if vec
                ]
                if points:
                    await self.client.upsert(collection_name=self.collection, points=points)
                async with state_lock:
                    consecutive_failures = 0
                    written += len(points)
                logger.info(f"[Ingest] Batch {index + 1}/{len(batches)} upserted "
                            f"{len(points)} (total {written}/{len(docs)}).")
                return "ok"

            outcomes = await asyncio.gather(
                *(_embed_and_upsert(b, i) for i, b in enumerate(batches)))
            skipped = outcomes.count("skipped")
            if skipped:
                logger.warning(f"[Ingest] {skipped} batches skipped after abort.")
        elif not stale_ids:
            logger.info("[Ingest] Catalog unchanged; nothing to embed.")
            return 0

        # Delete stale points ONLY after a fully clean run: during an id-scheme
        # migration a product's old point is "stale" while its replacement is
        # in the changed set — removing it before the new one is written (or
        # when embedding failed) would punch holes in the searchable catalog.
        if stale_ids:
            if failed_batches == 0:
                await self.client.delete(
                    collection_name=self.collection,
                    points_selector=qmodels.PointIdsList(
                        points=[int(pid) if pid.isdigit() else pid for pid in stale_ids]),
                )
                logger.info(f"[Ingest] Deleted {len(stale_ids)} stale points.")
            else:
                logger.warning(f"[Ingest] Keeping {len(stale_ids)} stale points until a clean "
                               f"sync ({failed_batches} batches failed this run).")

        logger.info(f"[Ingest] Sync done: {written} written, {failed_batches} batches failed.")
        return written

    # -------------------------------------------------------------- queries
    async def get_all_documents(self) -> List[Dict[str, Any]]:
        if not await self.ensure_ready():
            return []
        all_docs: List[Dict[str, Any]] = []
        next_offset = None
        while True:
            points, next_offset = await self.client.scroll(
                collection_name=self.collection, limit=256,
                with_payload=True, with_vectors=False, offset=next_offset)
            for p in points:
                if p.payload:
                    doc = dict(p.payload)
                    doc["id"] = str(p.id)
                    all_docs.append(doc)
            if next_offset is None:
                return all_docs

    async def search_dense(self, query_vector: List[float], limit: int = 12) -> List[Any]:
        if not await self.ensure_ready():
            return []
        res = await self.client.query_points(
            collection_name=self.collection, query=query_vector, limit=limit, with_payload=True)
        return list(res.points or [])

    async def search_by_exact_code(self, code: str) -> List[Any]:
        """Exact CodigoArticulo match via payload filter (no vector similarity)."""
        if not await self.ensure_ready():
            return []
        match_filter = qmodels.Filter(must=[
            qmodels.FieldCondition(key="CodigoArticulo", match=qmodels.MatchValue(value=code))])
        points, _ = await self.client.scroll(
            collection_name=self.collection, scroll_filter=match_filter,
            limit=5, with_payload=True, with_vectors=False)
        return points


qdrant_service = QdrantService()