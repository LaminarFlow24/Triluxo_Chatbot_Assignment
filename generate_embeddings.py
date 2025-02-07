import json
import faiss
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS

# Load courses.json
with open("courses.json", "r", encoding="utf-8") as f:
    courses = json.load(f)

# Create Document objects from each course
documents = []
for course in courses:
    content = (
        f"Course Name: {course.get('Course Name', '')}\n"
        f"Description: {course.get('Course Description', '')}\n"
        f"Curriculum: {', '.join(course.get('Course Curriculum', []))}\n"
        f"Price: {course.get('Course Price', 'N/A')}"
    )
    doc = Document(page_content=content, metadata=course)
    documents.append(doc)

# Create embeddings using a local SentenceTransformer model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Build a FAISS vector store from the documents
vectorstore = FAISS.from_documents(documents, embeddings)

# Save the FAISS index to disk
faiss.write_index(vectorstore.index, "courses_faiss.index")

# Save the metadata for each course using the original documents list.
metadata_list = [doc.metadata for doc in documents]
with open("courses_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, indent=4)

print("Embeddings generated and saved to 'courses_faiss.index' and 'courses_metadata.json'.")
