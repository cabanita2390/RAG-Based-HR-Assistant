import json
import re
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def evaluate_answer(
    question: str,
    answer: str,
    chunks: List[Dict[str, Any]],
    model_name: str = "gpt-4o-mini"
) -> Dict[str, Any]:

    llm = ChatOpenAI(model=model_name, temperature=0)

    evaluation_prompt = ChatPromptTemplate.from_template("""
Eres un evaluador técnico de sistemas RAG.

Evalúa la calidad de la respuesta usando estas dimensiones:

1. Relevancia de los chunks.
2. Fidelidad al contexto.
3. Completitud respecto a la pregunta.

Responde SOLO con un JSON válido.
No agregues texto adicional.
No uses bloques de código.
No agregues explicaciones fuera del JSON.

Formato obligatorio:

{{
  "score": entero entre 0 y 10,
  "reason": "explicación de al menos 50 caracteres"
}}

Pregunta:
{question}

Respuesta:
{answer}

Chunks:
{chunks}
""")

    chain = evaluation_prompt | llm

    response = chain.invoke({
        "question": question,
        "answer": answer,
        "chunks": json.dumps(chunks, ensure_ascii=False)
    })

    content = response.content.strip()

    # Extraer JSON aunque venga con texto adicional
    match = re.search(r"\{.*\}", content, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return {
        "score": 0,
        "reason": "El evaluador no devolvió un JSON válido."
    }