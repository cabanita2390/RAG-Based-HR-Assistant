import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
print("Intentando inicializar embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("Intentando conectar con Chroma...")
try:
    # Esto probará si el binario de Chroma/SQLite rompe el proceso
    vectordb = Chroma(
        collection_name="hr_faq",
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
    print("¡Éxito! Chroma cargó correctamente.")
    print("Conteo:", vectordb._collection.count())
except Exception as e:
    print(f"Error capturado: {e}")