from PyPDF2 import PdfReader
import docx
import pandas as pd
import logging


def extract_text_from_pdf(path):
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logging.error(f"Error leyendo PDF: {e}")
        return None


def extract_text_from_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logging.error(f"Error leyendo DOCX: {e}")
        return None


def extract_text_from_txt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error leyendo TXT: {e}")
        return None


def extract_text_from_csv(path):
    try:
        df = pd.read_csv(path)
        return df.to_string(index=False)
    except Exception as e:
        logging.error(f"Error leyendo CSV: {e}")
        return None


def extract_text_from_xlsx(path):
    try:
        df = pd.read_excel(path)
        return df.to_string(index=False)
    except Exception as e:
        logging.error(f"Error leyendo XLSX: {e}")
        return None
