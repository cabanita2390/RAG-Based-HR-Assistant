# RAG-RAG-FAQ-Assistant

## Descripción del Proyecto

RAG-FAQ-Assistant es un sistema de recuperación aumentada por generación (Retrieval-Augmented Generation, RAG) diseñado para automatizar la atención de preguntas frecuentes en una empresa SaaS de recursos humanos. El sistema procesa documentación interna no estructurada, la segmenta en fragmentos semánticamente manejables, genera embeddings vectoriales y los almacena en una base de datos persistente (ChromaDB).

Ante una consulta del usuario, el sistema recupera los fragmentos más relevantes mediante búsqueda por similitud coseno y utiliza un modelo de lenguaje para generar respuestas fundamentadas exclusivamente en el contexto recuperado. La salida se entrega en formato JSON estructurado para garantizar transparencia, trazabilidad y auditabilidad.

Arquitectura del Sistema

El sistema implementa una arquitectura RAG en dos pipelines desacoplados:

1️⃣ Pipeline de Indexación

* Carga del documento fuente (faq_document.txt)
* Segmentación mediante RecursiveCharacterTextSplitter
* chunk_size = 300
* chunk_overlap = 50
* Generación de embeddings con text-embedding-3-small
* Almacenamiento persistente en ChromaDB (./chroma_db)
Este pipeline permite reconstruir completamente el índice vectorial desde el documento original.

2️⃣ Pipeline de Consulta

* Conversión de la pregunta del usuario en embedding
* Búsqueda vectorial por similitud coseno (top-k = 3)
* Recuperación de los fragmentos más relevantes
* Ensamblado de contexto
* Generación de respuesta con gpt-4o-mini
* Retorno en formato JSON estructurado
El sistema devuelve:

´´ bash
{
  "user_question": "...",
  "system_answer": "...",
  "chunks_related": [
    {
      "chunk_id": "...",
      "text": "...",
      "similarity_score": 0.91
    }
  ]
}
´´

Justificación Técnica
Estrategia de Chunking

Se utiliza segmentación por tamaño fijo con solapamiento (300 / 50) para equilibrar coherencia contextual y control del límite de tokens. Esta estrategia reduce la fragmentación semántica y asegura que cada chunk permanezca dentro del rango óptimo de procesamiento del modelo.

Método de Búsqueda Vectorial

Se implementa búsqueda por similitud coseno sobre embeddings generados con OpenAI. Este método permite recuperar fragmentos semánticamente relevantes incluso cuando la formulación de la pregunta difiere del texto original.

Uso de RAG

El sistema aplica recuperación antes de generación, lo que permite:

Incorporar conocimiento actualizado sin reentrenamiento.

Minimizar alucinaciones.

Proporcionar transparencia al exponer los fragmentos utilizados.



Instalación

Clonar el repositorio:
git clone <repo-url>
cd RAG-FAQ-Assistant

