import os
import urllib
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()  # Carga variables desde .env


def get_sqlserver_session():
    user = os.getenv("SQLSERVER_USER")
    password = urllib.parse.quote_plus(os.getenv("SQLSERVER_PASSWORD"))
    host = os.getenv("SQLSERVER_HOST")
    db = os.getenv("SQLSERVER_DB")

    conn_str = f"mssql+pyodbc://{user}:{password}@{host}/{db}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(conn_str)
    return sessionmaker(bind=engine)()


def get_sqlite_session():
    path = os.getenv("SQLITE_PATH", "./db.sqlite3")
    engine = create_engine(f"sqlite:///{path}")
    return sessionmaker(bind=engine)()


def get_postgres_session():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")

    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(conn_str)
    return sessionmaker(bind=engine)()
