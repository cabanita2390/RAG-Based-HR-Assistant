# RAG-Based HR Assistant

## Descripción del Proyecto

RAG-Based HR Assistant es un sistema de Retrieval-Augmented Generation (RAG) diseñado para automatizar la atención de preguntas frecuentes en una empresa SaaS de Recursos Humanos.

El sistema procesa documentación interna no estructurada, la segmenta en fragmentos semánticamente coherentes, genera embeddings vectoriales mediante OpenAI y los almacena en una base de datos vectorial persistente (ChromaDB).

Ante una consulta, el sistema:

1. Convierte la pregunta en embedding.
2. Recupera los fragmentos más relevantes mediante búsqueda vectorial.
3. Construye un contexto fundamentado.
4. Genera una respuesta usando un LLM.
5. Devuelve un JSON estructurado con trazabilidad completa.

El sistema evita alucinaciones respondiendo exclusivamente con información presente en los documentos indexados.

## Arquitectura del Sistema

El sistema implementa una arquitectura RAG desacoplada en dos pipelines independientes.

### 1️⃣ Pipeline de Indexación (build_index.py)

**Etapas:**

- Carga del documento fuente (`data/faq_document.txt`)
- Segmentación del texto usando `RecursiveCharacterTextSplitter`:
  - `chunk_size` = 300
  - `chunk_overlap` = 50
- Generación de embeddings con:
  - `text-embedding-3-small`
- Almacenamiento persistente en ChromaDB:
  - Directorio: `./chroma_db`
  - Colección: `hr_faq`

*La persistencia es automática en versiones modernas de ChromaDB (no se requiere `.persist()`).*

Este pipeline permite reconstruir completamente el índice vectorial desde cero.

### 2️⃣ Pipeline de Consulta (query.py)

**Etapas:**

- Conversión de la pregunta del usuario en embedding (misma dimensionalidad que los chunks).
- Búsqueda vectorial k-NN sobre ChromaDB.
- Recuperación de los top-k fragmentos más cercanos.
- Ordenamiento por distancia coseno (menor distancia = mayor similitud).
- Ensamblado de contexto.
- Generación de respuesta con `gpt-4o-mini`.
- Retorno en JSON estructurado.

## Método de Búsqueda Vectorial

Se utiliza:

- k-Nearest Neighbors (k-NN)
- Métrica: cosine distance (implementada por ChromaDB)

**Importante:**

ChromaDB devuelve cosine distance, no cosine similarity.

Por lo tanto:

- Menor distancia → mayor similitud.
- Los resultados se ordenan en orden ascendente.

Se utiliza `top_k = 5` para mejorar recall y robustez ante ambigüedades.

## Formato de Salida

El sistema devuelve un JSON estructurado con trazabilidad completa:

```json
{
  "user_question": "¿Cuántos días de vacaciones tengo?",
  "system_answer": "Si eres empleado de tiempo completo...",
  "chunks_related": [
    {
      "chunk_id": 6,
      "text": "...",
      "similarity_score": 0.0021
    }
  ]
}
```

Esto permite:

- Auditoría
- Transparencia
- Verificación de grounding
- Integración sencilla con frontend o API

## Justificación Técnica

### Estrategia de Chunking

Se utiliza segmentación por tamaño fijo con solapamiento (300 / 50) para:

- Preservar coherencia semántica.
- Evitar fragmentación excesiva.
- Mantener cada chunk dentro del rango óptimo de tokens.
- Reducir pérdida de contexto entre fragmentos contiguos.

El solapamiento mejora recuperación en preguntas que cruzan límites de fragmento.

### Embeddings

Se utilizan embeddings de OpenAI (`text-embedding-3-small`) por:

- Alta calidad semántica.
- Robustez en parafraseo.
- Bajo costo relativo.
- Dimensionalidad adecuada para recuperación eficiente.

### Uso de RAG

El sistema aplica recuperación antes de generación, lo que permite:

- Incorporar conocimiento privado sin fine-tuning.
- Actualizar documentos sin reentrenar modelos.
- Minimizar alucinaciones.
- Proporcionar trazabilidad mediante chunks relacionados.

## Requisitos del Entorno

Python recomendado:

- Python 3.11.x

*Versiones más recientes (3.13+ / 3.14) pueden generar conflictos binarios con dependencias de ChromaDB y ONNX.*

## Instalación

Clonar repositorio:

```bash
git clone <repo-url>
cd RAG-Based-HR-Assistant
```

Crear entorno virtual con Python 3.11:

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate
```

Instalar dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Configurar variables de entorno:

Crear archivo `.env`:

```env
OPENAI_API_KEY=your-key-here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
CHROMA_DIR=./chroma_db
CHROMA_COLLECTION=hr_faq
TOP_K=5
```

## Ejecución

### Construir índice

```bash
python src/build_index.py
```

### Ejecutar consulta

```bash
python src/query.py --question "¿Cuántos días de vacaciones tengo?"
```

## Estructura del Proyecto

```text
RAG-Based-HR-Assistant/
│
├── data/
│   └── faq_document.txt
│
├── src/
│   ├── build_index.py
│   └── query.py
│
├── outputs/
│   └── sample_queries.json
│
├── chroma_db/
├── requirements.txt
├── .env.example
└── README.md
```

🔬 Justificación Técnica de Diseño
1. Uso de Embeddings (OpenAI text-embedding-3-small)

Se seleccionó este modelo por:

Alta calidad semántica.

Bajo costo relativo.

Buen balance entre performance y dimensionalidad.

Compatibilidad nativa con LangChain.

Alternativas como Sentence-Transformers fueron consideradas, pero se priorizó estabilidad industrial y consistencia en evaluación.

2. Base de Datos Vectorial: Chroma (persistente)

Se eligió Chroma por:

Integración directa con LangChain.

Persistencia en disco.

Facilidad de uso para entornos locales.

Adecuado para prototipos industriales escalables.

La persistencia garantiza reproducibilidad y evita re-indexar en cada ejecución.

3. Métrica de Recuperación: Similitud Coseno

Chroma utiliza distancia coseno como métrica base.

En el sistema:

Se convierte distancia → similitud.

Se documenta explícitamente en la salida JSON.

Se usa top-k retrieval.

Esto cumple el requisito explícito del ejercicio académico.

4. Evaluador Automático LLM-Based

Se implementó un segundo LLM para evaluar:

Relevancia de los chunks.

Fidelidad al contexto.

Completitud de la respuesta.

Esto transforma el sistema en un RAG auditable y confiable, alineado con estándares industriales.