import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collections = client.list_collections()
print("Colecciones:", collections)