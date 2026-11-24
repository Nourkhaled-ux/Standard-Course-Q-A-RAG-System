# Standard Course Q&A RAG System

An end-to-end Retrieval-Augmented Generation (RAG) system that enables students to ask questions and receive accurate, context-aware answers based on course materials.

The system supports multiple file formats, including PDF, CSV, DOCX, and TXT, across three different courses. It extracts and processes the content, splits it into manageable chunks, generates embeddings, and stores them in a vector database for efficient semantic retrieval.

When a student submits a question, the retriever searches for the most relevant course content and provides it to a Large Language Model (LLM) to generate an answer grounded in the retrieved materials, with source attribution.

### Key Features
- Multi-course knowledge base supporting three courses.
- Support for PDF, CSV, DOCX, and TXT files.
- Text extraction, preprocessing, and chunking.
- Semantic embeddings for document representation.
- Vector database for efficient similarity search.
- Retrieval pipeline connected to an LLM.
- Context-grounded answers with source attribution.
- Interactive interface built with Streamlit or Gradio.

### RAG Pipeline
1. **Document Loading:** Load course materials from different file formats.
2. **Text Processing:** Extract, clean, and split documents into chunks.
3. **Embedding Generation:** Convert text chunks into vector representations.
4. **Vector Storage:** Store embeddings in a vector database.
5. **Retrieval:** Retrieve the most relevant chunks based on the student's question.
6. **Answer Generation:** Use an LLM to generate answers using the retrieved context.
7. **Source Attribution:** Display the sources supporting the generated answers.

### Objective
To provide students with an interactive course assistant that makes learning materials easier to search, understand, and access through natural-language questions.
