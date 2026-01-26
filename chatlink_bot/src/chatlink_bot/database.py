import os
import urllib.parse
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

# --- Postgres Configuration (Write Access) ---
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "password")
PG_HOST = os.getenv("POSTGRES_HOST", "db")
PG_DB = os.getenv("POSTGRES_DB", "chatlink")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")

PG_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

pg_engine = create_async_engine(PG_URL, echo=False)
AsyncSessionPG = async_sessionmaker(pg_engine, expire_on_commit=False, class_=AsyncSession)

# --- SQL Server Configuration (Read Access) ---
SQL_USER = os.getenv("SQLSERVER_USER")
SQL_PASS = os.getenv("SQLSERVER_PASSWORD")
SQL_HOST = os.getenv("SQLSERVER_HOST")
SQL_DB = os.getenv("SQLSERVER_DB")

# Construct ODBC Connection String
# Driver must match what is installed in Dockerfile (ODBC Driver 17 or 18)
# Since you have msodbcsql17 deb file, we assume Driver 17.
params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SQL_HOST};"
    f"DATABASE={SQL_DB};"
    f"UID={SQL_USER};"
    f"PWD={SQL_PASS};"
    f"TrustServerCertificate=yes;"
)

SQLSERVER_URL = f"mssql+aioodbc:///?odbc_connect={params}"

sql_engine = create_async_engine(SQLSERVER_URL, echo=False)
AsyncSessionSQL = async_sessionmaker(sql_engine, expire_on_commit=False, class_=AsyncSession)


class PGBase(DeclarativeBase):
    pass

class MSBase(DeclarativeBase):
    pass

# Dependency for FastAPI endpoints (Default Postgres)
async def get_db():
    async with AsyncSessionPG() as session:
        yield session