import os
import re
import hashlib
import chromadb
import pandas as pd
from pypdf import PdfReader
from docx import Document
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

DATA_FOLDER = "data"
DB_FOLDER = "chroma_db_gemini"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768

CHUNK_SIZE = 350
CHUNK_OVERLAP = 60
EMBEDDING_BATCH_SIZE = 32


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Set your API key before running ingest.py."
    )

gemini_client = genai.Client(api_key=api_key)

client = chromadb.PersistentClient(
    path=DB_FOLDER
)

collection = client.get_or_create_collection(
    name="course_materials",
    metadata={"hnsw:space": "cosine"}
)


def clean_text(text):

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_pdf(file_path):

    documents = []

    reader = PdfReader(file_path)

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        text = clean_text(text)

        if text:

            documents.append({
                "text": text,
                "locator": f"Page {page_number}"
            })

    return documents

def extract_docx(file_path):

    doc = Document(file_path)

    documents = []

    for paragraph_number, paragraph in enumerate(
        doc.paragraphs,
        start=1
    ):

        text = clean_text(paragraph.text)

        if text:

            documents.append({
                "text": text,
                "locator": f"Paragraph {paragraph_number}"
            })

    for table_number, table in enumerate(
        doc.tables,
        start=1
    ):

        for row_number, row in enumerate(
            table.rows,
            start=1
        ):

            row_text = " | ".join(
                clean_text(cell.text)
                for cell in row.cells
            )

            row_text = clean_text(row_text)

            if row_text:

                documents.append({
                    "text": row_text,
                    "locator": (
                        f"Table {table_number}, "
                        f"Row {row_number}"
                    )
                })

    return documents


def extract_csv(file_path):

    df = pd.read_csv(
        file_path,
        encoding="utf-8",
        on_bad_lines="skip"
    )

    documents = []

    for row_number, row in df.iterrows():

        row_text = " | ".join(
            f"{column}: {value}"
            for column, value in row.items()
            if pd.notna(value)
        )

        row_text = clean_text(row_text)

        if row_text:

            documents.append({
                "text": row_text,
                "locator": f"Row {row_number + 2}"
            })

    return documents


def extract_txt(file_path):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

    except UnicodeDecodeError:

        with open(
            file_path,
            "r",
            encoding="latin-1"
        ) as file:

            text = file.read()

    text = clean_text(text)

    if text:

        return [{
            "text": text,
            "locator": "Text file"
        }]

    return []


def extract_file(file_path):

    extension = os.path.splitext(
        file_path
    )[1].lower()

    if extension == ".pdf":

        return extract_pdf(file_path)

    elif extension == ".docx":

        return extract_docx(file_path)

    elif extension == ".csv":

        return extract_csv(file_path)

    elif extension == ".txt":

        return extract_txt(file_path)

    else:

        return []

def split_into_chunks(text):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        if end >= len(words):

            break

        start = end - CHUNK_OVERLAP

    return chunks

def create_embeddings(texts):

    all_embeddings = []

    for start in range(
        0,
        len(texts),
        EMBEDDING_BATCH_SIZE
    ):

        batch = texts[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        result = gemini_client.models.embed_content(

            model=EMBEDDING_MODEL,

            contents=batch,

            config=types.EmbedContentConfig(

                task_type="RETRIEVAL_DOCUMENT",

                output_dimensionality=EMBEDDING_DIMENSION
            )
        )

        embeddings = [
            item.values
            for item in result.embeddings
        ]

        all_embeddings.extend(embeddings)

    return all_embeddings

def process_courses():

    if not os.path.exists(DATA_FOLDER):

        print("Data folder does not exist!")

        return

    total_chunks = 0

    for course_name in sorted(
        os.listdir(DATA_FOLDER)
    ):

        course_path = os.path.join(
            DATA_FOLDER,
            course_name
        )

        if not os.path.isdir(course_path):

            continue

        print(
            f"\nProcessing course: {course_name}"
        )

        for file_name in sorted(
            os.listdir(course_path)
        ):

            file_path = os.path.join(
                course_path,
                file_name
            )

            if not os.path.isfile(file_path):

                continue

            print(
                f"Reading: {file_name}"
            )

            try:

                extracted_documents = extract_file(
                    file_path
                )

                all_chunks = []

                all_ids = []

                all_metadatas = []

                for document in extracted_documents:

                    chunks = split_into_chunks(
                        document["text"]
                    )

                    for chunk_index, chunk in enumerate(
                        chunks
                    ):

                        chunk_id = hashlib.sha256(

                            (
                                f"{course_name}|"
                                f"{file_name}|"
                                f"{document['locator']}|"
                                f"{chunk_index}|"
                                f"{chunk}"
                            ).encode("utf-8")

                        ).hexdigest()

                        all_chunks.append(chunk)

                        all_ids.append(chunk_id)

                        all_metadatas.append({

                            "course": course_name,

                            "source": file_name,

                            "locator": document["locator"]

                        })

                if not all_chunks:

                    print(
                        f"No readable text found in {file_name}"
                    )

                    continue

                embeddings = create_embeddings(
                    all_chunks
                )

                collection.upsert(

                    ids=all_ids,

                    documents=all_chunks,

                    embeddings=embeddings,

                    metadatas=all_metadatas

                )

                total_chunks += len(all_chunks)

                print(
                    f"Added {len(all_chunks)} chunks "
                    f"from {file_name}"
                )

            except Exception as error:

                print(
                    f"Error processing {file_name}: {error}"
                )

    print("\n================================")

    print("Data processing completed!")

    print(
        f"Chunks processed: {total_chunks}"
    )

    print(
        f"Total database records: {collection.count()}"
    )

    print("================================")

if __name__ == "__main__":

    process_courses()
