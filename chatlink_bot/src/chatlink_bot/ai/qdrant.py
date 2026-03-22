# src/chatlink_bot/ai/qdrant.py
import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import AsyncQdrantClient, models as qmodels
from sqlalchemy import select

from cudara_client import CudaraClient

from ..database import AsyncSessionSQL
from ..models import MSArticle

logger = logging.getLogger("Qdrant")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_FALLBACK_URL = os.getenv("QDRANT_FALLBACK_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "products")
QDRANT_RESET_ON_STARTUP = os.getenv("QDRANT_RESET_ON_STARTUP", "false").lower() == "true"

CUDARA_URL = os.getenv("CUDARA_URL", "http://cudara:8000")
EMBED_MODEL = os.getenv("CUDARA_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class QdrantService:
    def __init__(self) -> None:
        self.primary_url = QDRANT_URL
        self.fallback_url = QDRANT_FALLBACK_URL
        
        self.client = AsyncQdrantClient(url=self.primary_url, prefer_grpc=False)
        self.collection = QDRANT_COLLECTION

        self._collection_ready = False
        self._lock = asyncio.Lock()
        self._using_fallback = False

        self._cudara = CudaraClient(CUDARA_URL)

    async def health_check(self) -> Dict[str, Any]:
        """
        Returns status dict. Uses get_collections() to verify connectivity 
        without erroring if the specific collection is missing.
        """
        try:
            col_res = await self.client.get_collections()
            
            # Check if our collection exists in the list
            exists = any(c.name == self.collection for c in col_res.collections)
            
            return {
                "status": "ok",
                "connected": True,
                "collection_exists": exists,
                "total_collections": len(col_res.collections),
                "url": self.fallback_url if self._using_fallback else self.primary_url
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "url": self.fallback_url if self._using_fallback else self.primary_url
            }

    async def _detect_vector_size(self) -> int:
        # Retry loop: wait up to 300s for the model to load
        max_retries = 30
        for i in range(max_retries):
            try:
                resp = await asyncio.to_thread(self._cudara.embed, EMBED_MODEL, ["vector_size_probe"])
                vec = []
                if hasattr(resp, "embeddings") and resp.embeddings:
                    vec = resp.embeddings[0]
                elif hasattr(resp, "embedding") and resp.embedding:
                    vec = resp.embedding
                
                if vec:
                    return len(vec)
            except Exception as e:
                # If it's the specific "not ready" error or connection error, wait
                logger.warning(f"Waiting for embedding model {EMBED_MODEL} ({i+1}/{max_retries})... Error: {e}")
            
            await asyncio.sleep(10)

        raise RuntimeError(f"Could not detect vector size. Model {EMBED_MODEL} is unavailable after 300s.")

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
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                    on_disk=True,
                ),
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
        """
        Retorna el número de puntos (productos) actualmente en la colección.
        """
        if not await self.ensure_ready():
            return 0
        try:
            res = await self.client.count(collection_name=self.collection)
            return int(res.count)
        except Exception as e:
            logger.warning(f"Failed to get collection count: {e}")
            return 0

    @staticmethod
    def _build_product_text(a: MSArticle) -> str:
        # [MODIFICADO] Incluimos el CodigoArticulo al principio para mejorar el match vectorial parcial
        parts = [
            str(a.CodigoArticulo or "").strip(),
            a.DescripcionArticulo or "",
            a.Descripcion2Articulo or "",
            a.DescripcionLinea or "",
            a.ComentarioArticulo or "",
            a.MarcaProducto or "",
        ]
        return "\n".join(p.strip() for p in parts if p and p.strip())

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = await asyncio.to_thread(self._cudara.embed, model=EMBED_MODEL, input=texts)
        # It's now a dictionary
        return resp.get("embeddings", [])
    
    @staticmethod
    def _to_point_id(codigo_empresa: Any, codigo_articulo: Any) -> int | str:
        code = (str(codigo_articulo or "")).strip()
        if code.isdigit():
            # Qdrant accepts unsigned integers -> send as int (NOT string)
            return int(code)
        # fallback for alphanumeric codes: deterministic UUID
        key = f"{str(codigo_empresa or '').strip()}:{code}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    async def ingest_products_from_sqlserver(self, batch_size: int = 64) -> int:
        """
        NOTES.md requirement:
          - Startup + Daily
          - Only Articulos where K_BOT = 0
        """
        if not await self.ensure_ready():
            return 0

        total = 0
        offset = 0

        while True:
            async with AsyncSessionSQL() as s:
                stmt = (
                    select(MSArticle)
                    .where(MSArticle.K_BOT == 0)  # strict NOTES filter
                    .order_by(MSArticle.CodigoArticulo)
                    .offset(offset)
                    .limit(batch_size)
                )
                res = await s.execute(stmt)
                rows = res.scalars().all()

            if not rows:
                break

            payloads: List[Dict[str, Any]] = []
            texts: List[str] = []
            ids: List[str] = []

            for a in rows:
                content = self._build_product_text(a)
                if not content:
                    continue
                pid = self._to_point_id(a.CodigoEmpresa, a.CodigoArticulo)
                ids.append(pid)
                texts.append(content)
                payloads.append(
                    {
                        "CodigoArticulo": str(a.CodigoArticulo).strip(),
                        "DescripcionArticulo": a.DescripcionArticulo,
                        "Descripcion2Articulo": a.Descripcion2Articulo,
                        "DescripcionLinea": a.DescripcionLinea,
                        "ComentarioArticulo": a.ComentarioArticulo,
                        "MarcaProducto": a.MarcaProducto,
                        "K_BOT": int(a.K_BOT or -1),
                        "content": content,
                    }
                )

            if not ids:
                offset += batch_size
                continue

            embeddings = await self._embed_batch(texts)
            points: List[qmodels.PointStruct] = []
            for pid, vec, payload in zip(ids, embeddings, payloads):
                if not vec:
                    continue
                points.append(qmodels.PointStruct(id=pid, vector=vec, payload=payload))

            if points:
                await self.client.upsert(collection_name=self.collection, points=points)
                total += len(points)
                logger.info(f"Upserted {len(points)} products into Qdrant.")

            offset += batch_size

        logger.info(f"Product ingestion completed. Total upserted: {total}")
        return total

    async def get_all_documents(self) -> List[Dict[str, Any]]:
        if not await self.ensure_ready():
            return []

        all_docs: List[Dict[str, Any]] = []
        next_offset: Optional[int] = None

        while True:
            points, next_offset = await self.client.scroll(
                collection_name=self.collection,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )
            for p in points:
                if p.payload:
                    doc = dict(p.payload)
                    doc["id"] = str(p.id)
                    all_docs.append(doc)
            if next_offset is None:
                break

        return all_docs

    async def search_dense(self, query_vector: List[float], limit: int = 12) -> List[Any]:
        if not await self.ensure_ready():
            return []
        res = await self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return list(res.points or [])
    
    async def search_by_exact_code(self, code: str) -> List[Any]:
        """
        [NUEVO] Busca explícitamente por CodigoArticulo usando un filtro de metadatos.
        Ignora similitud vectorial, busca coincidencia exacta de valor.
        """
        if not await self.ensure_ready():
            return []
        
        # Filtro estricto: payload.CodigoArticulo == code
        match_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="CodigoArticulo",
                    match=qmodels.MatchValue(value=code)
                )
            ]
        )
        
        points, _ = await self.client.scroll(
            collection_name=self.collection,
            scroll_filter=match_filter,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        return points


qdrant_service = QdrantService()