import os
from flask import Flask, request, jsonify, send_from_directory
from flask_restful import Resource, Api
from flask_cors import CORS
import json

# Import LangChain components
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain

# Import Hugging Face Pipeline LLM wrapper from LangChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --- Flask Setup ---
app = Flask(__name__, static_folder='static')
CORS(app)
api = Api(app)

# --- Load courses.json ---
with open("courses.json", "r", encoding="utf-8") as f:
    courses = json.load(f)

# --- Create Documents from Course Data ---
documents = []
for course in courses:
    # Build content for embeddings (this content is not directly displayed)
    content = (
        f"Course Name: {course.get('Course Name', '')}\n"
        f"Description: {course.get('Course Description', '')}\n"
        f"Curriculum: {', '.join(course.get('Course Curriculum', []))}\n"
        f"Price: {course.get('Course Price', 'N/A')}"
    )
    doc = Document(page_content=content, metadata=course)
    documents.append(doc)

# --- Create Embeddings and Build the FAISS Vector Store ---
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# --- Create the Hugging Face LLM for Chatbot Responses ---
model_name = "google/flan-t5-base"  # A free, reasonably accurate model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    truncation=True
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# --- Build the Conversational Retrieval Chain ---
chatbot_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

# --- Parameters for Course Filtering ---
MAX_COURSES = 5
DISTANCE_THRESHOLD = 1.0  # Adjust based on your embedding space

# --- Define Synonyms for "Course" ---
COURSE_SYNONYMS = {
    "course", "courses", "programme", "program", "session", "class", "classes",
    "training", "education", "learn", "learn more", "know more", "find out", "explore", "information"
}

# --- Define the Chatbot Resource ---
class Chatbot(Resource):
    def post(self):
        data = request.get_json()
        user_message = data.get("message", "").strip()
        history = data.get("history", [])  # Expected as a list of [question, answer] pairs

        # Convert history (list-of-lists) into a list of tuples for internal processing.
        chat_history = [tuple(item) for item in history]

        # --- Basic Questions Handling ---
        basic_questions = {
            "who are you?": "I am a course chatbot built using LangChain and a free Hugging Face model. I can help you with information about our courses.",
            "who are you": "I am a course chatbot built using LangChain and a free Hugging Face model. I can help you with information about our courses.",
            "what is your name?": "I am your friendly course chatbot!",
            "what is your name": "I am your friendly course chatbot!"
        }
        if user_message.lower() in basic_questions:
            answer = basic_questions[user_message.lower()]
            chat_history.append((user_message, answer))
            history = [list(pair) for pair in chat_history]
            return jsonify({"response": answer, "history": history})

        # --- Determine if Course Retrieval Should be Triggered ---
        user_message_lower = user_message.lower()
        if any(syn in user_message_lower for syn in COURSE_SYNONYMS):
            results_with_scores = vectorstore.similarity_search_with_score(user_message, k=10)
            filtered_results = [(doc, score) for doc, score in results_with_scores if score <= DISTANCE_THRESHOLD]
            filtered_results = sorted(filtered_results, key=lambda x: x[1])[:MAX_COURSES]

            if not filtered_results:
                answer = "Sorry, I couldn't find any courses matching that query."
            else:
                # Check for specific field requests; if none, show all details.
                requested_fields = []
                if "curriculum" in user_message_lower:
                    requested_fields.append("curriculum")
                if "price" in user_message_lower:
                    requested_fields.append("price")
                if "link" in user_message_lower:
                    requested_fields.append("link")
                if "description" in user_message_lower:
                    requested_fields.append("description")
                show_all = len(requested_fields) == 0

                answer_lines = [
                    "<div style='font-weight:bold;margin-bottom:10px;'>Here are the relevant courses:</div>"
                ]
                for doc, score in filtered_results:
                    meta = doc.metadata
                    course_name = meta.get("Course Name", "N/A")
                    description = meta.get("Course Description", "N/A")
                    link = meta.get("Course Link", "N/A")
                    curriculum = meta.get("Course Curriculum", "N/A")
                    price = meta.get("Course Price", "N/A")
                    lessons = meta.get("Number of Lessons", "N/A")
                    if isinstance(curriculum, list):
                        curriculum = ", ".join(curriculum)

                    # Compute Total Price = price * number of lessons
                    try:
                        # Remove any "$" from price, then convert to float
                        p = float(str(price).replace("$", "").strip())
                        n = float(str(lessons).strip())
                        total_price = p * n
                        total_price_str = "$" + "{:.2f}".format(total_price)
                    except Exception:
                        total_price_str = "N/A"

                    # Build an organized HTML card for the course.
                    course_info = f"""
                    <div class="course-card">
                      <h2><strong>{course_name.upper()}</strong></h2>
                      <p><strong>Link:</strong> <a href="{link}" target="_blank">{link}</a></p>
                      <p><strong>Number of Lessons:</strong> {lessons}</p>
                      <p><strong>Total Price:</strong> {total_price_str}</p>
                    """
                    if show_all or "description" in requested_fields:
                        course_info += f"<p><strong>Description:</strong> {description}</p>"
                    if show_all or "price" in requested_fields:
                        course_info += f"<p><strong>Price:</strong> {price}</p>"
                    if show_all or "curriculum" in requested_fields:
                        course_info += f"<p><strong>Curriculum:</strong> {curriculum}</p>"
                    course_info += f"""
                      <p>
                        <strong>Actions:</strong> 
                        <a href="https://brainlox.com/book-free-demo" target="_blank">Book a Free Demo</a> | 
                        <a href="https://brainlox.com/contact" target="_blank">Enquire Now</a>
                      </p>
                      <p><em>Score:</em> {score:.2f}</p>
                    </div>
                    """
                    answer_lines.append(course_info)
                answer = "\n".join(answer_lines)
            chat_history.append((user_message, answer))
            history = [list(pair) for pair in chat_history]
            return jsonify({"response": answer, "history": history})

        # --- For Other Queries, Use the Conversational Retrieval Chain ---
        result = chatbot_chain({"question": user_message, "chat_history": chat_history})
        answer = result.get("answer", "Sorry, I couldn't generate an answer at this time.")
        chat_history.append((user_message, answer))
        history = [list(pair) for pair in chat_history]
        return jsonify({"response": answer, "history": history})

# --- Register the Chatbot Resource at /chat ---
api.add_resource(Chatbot, '/chat')

# --- Serve the Static HTML Page at the Root URL ---
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
