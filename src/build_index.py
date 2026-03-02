import os
import shutil
from dataclasses import dataclass
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


@dataclass(frozen=True)
class IndexConfig:
    faq_path: str
    persist_dir: str
    collection_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


def load_config() -> IndexConfig:
    load_dotenv()

    faq_path = os.getenv("FAQ_PATH", "data/faq_document.txt")
    persist_dir = os.getenv("CHROMA_DIR", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION", "hr_faq")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    chunk_size = int(os.getenv("CHUNK_SIZE", "300"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE debe ser > 0")
    if chunk_overlap < 0:
        raise ValueError("CHUNK_OVERLAP debe ser >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP debe ser < CHUNK_SIZE")

    return IndexConfig(
        faq_path=faq_path,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def read_text_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    # Manejo básico de codificación: intenta utf-8, luego latin-1
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Creamos documentos a partir de un solo texto
    docs = splitter.create_documents([text])

    # Añadimos metadata útil (chunk_id se agregará luego con ids explícitos en Chroma)
    for i, d in enumerate(docs):
        d.metadata.update({"chunk_index": i})

    return docs


def ensure_min_chunks(docs: List[Document], min_chunks: int = 20) -> None:
    if len(docs) < min_chunks:
        raise ValueError(
            f"Se generaron {len(docs)} chunks (< {min_chunks}). "
            f"Aumenta el tamaño del documento o reduce CHUNK_SIZE."
        )


def reset_persist_dir(persist_dir: str) -> None:
    """
    Para reproducibilidad: borra el directorio de Chroma si existe y lo recrea.
    En un flujo más productivo, podrías versionar o migrar colecciones en lugar de borrar.
    """
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)


def build_and_persist_index(
    docs: List[Document],
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
) -> Tuple[Chroma, List[str]]:
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # IDs estables: chunk_0000, chunk_0001, ...
    ids = [f"chunk_{i:04d}" for i in range(len(docs))]

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        ids=ids,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    # En algunas versiones, persist() es opcional, pero lo llamamos explícitamente.
    return vectordb, ids


def main() -> None:
    cfg = load_config()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Falta OPENAI_API_KEY. Configúrala en tu entorno o en un .env."
        )

    print("[1/4] Leyendo documento...")
    text = read_text_file(cfg.faq_path)

    print("[2/4] Generando chunks...")
    docs = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
    ensure_min_chunks(docs, min_chunks=20)
    print(f"  - Chunks generados: {len(docs)}")

    print("[3/4] Reiniciando directorio persistente...")
    reset_persist_dir(cfg.persist_dir)

    print("[4/4] Generando embeddings y guardando en Chroma (persistente)...")
    vectordb, ids = build_and_persist_index(
        docs=docs,
        persist_dir=cfg.persist_dir,
        collection_name=cfg.collection_name,
        embedding_model=cfg.embedding_model,
    )

    # Verificación rápida
    count = vectordb._collection.count()
    print(f"  - Colección: {cfg.collection_name}")
    print(f"  - Persist dir: {cfg.persist_dir}")
    print(f"  - Documentos indexados: {count}")
    print(f"  - Ejemplo ID: {ids[0]}")


if __name__ == "__main__":
    main()