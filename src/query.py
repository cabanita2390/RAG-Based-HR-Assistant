import os
import json
import argparse
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma


def load_environment():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Falta OPENAI_API_KEY en el entorno.")

    return {
        "persist_dir": os.getenv("CHROMA_DIR", "./chroma_db"),
        "collection_name": os.getenv("CHROMA_COLLECTION", "hr_faq"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "top_k": int(os.getenv("TOP_K", "3")),
    }



import chromadb


def load_vectorstore(persist_dir: str, collection_name: str, embedding_model: str):
    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Cliente explícito de Chroma persistente
    client = chromadb.PersistentClient(path=persist_dir)

    vectordb = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    count = vectordb._collection.count()
    print("DEBUG: Documentos en colección:", count)

    if count == 0:
        raise ValueError(
            f"La colección '{collection_name}' está vacía o no fue cargada correctamente."
        )

    return vectordb

def retrieve_chunks(vectordb, question: str, top_k: int):
    results = vectordb.similarity_search_with_score(question, k=top_k)

    chunks = []
    for doc, score in results:
        chunks.append({
            "chunk_id": doc.metadata.get("chunk_index"),
            "text": doc.page_content,
            "similarity_score": round(score, 4)
        })

    return chunks


def build_context(chunks: List[Dict[str, Any]]) -> str:
    return "\n\n".join(chunk["text"] for chunk in chunks)


def generate_answer(llm_model: str, question: str, context: str) -> str:
    llm = ChatOpenAI(model=llm_model, temperature=0)

    prompt = ChatPromptTemplate.from_template("""
Responde la pregunta utilizando únicamente la información presente en el contexto.

Si existen múltiples escenarios posibles (por ejemplo, empleados de tiempo completo o medio tiempo),
indica claramente cada caso con su respectiva respuesta.

Si la información no está en el contexto, responde exactamente:
"No encontre esa informacion en los documentos."

Contexto:
{context}

Pregunta:
{question}
""")

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content.strip()


def main():

    try:
        print("DEBUG: Entrando a main()")
        parser = argparse.ArgumentParser()
        parser.add_argument("--question", type=str, required=True)
        args = parser.parse_args()

        print("DEBUG: Cargando entorno")
        config = load_environment()

        print("DEBUG: Cargando vectorstore")
        vectordb = load_vectorstore(
            config["persist_dir"],
            config["collection_name"],
            config["embedding_model"],
        )

        print("DEBUG: Obteniendo chunks")
        chunks = retrieve_chunks(vectordb, args.question, config["top_k"])
        chunks = sorted(chunks, key=lambda x: x["similarity_score"])
        context = build_context(chunks)

        print("DEBUG: Generando respuesta")
        answer = generate_answer(
            config["llm_model"],
            args.question,
            context
        )

        result = {
            "user_question": args.question,
            "system_answer": answer,
            "chunks_related": chunks
        }

        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Error al ejecutar el script: {str(e)}")


if __name__ == "__main__":
    main()