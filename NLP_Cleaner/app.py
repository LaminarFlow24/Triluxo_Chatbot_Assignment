from flask import Flask, request, jsonify, send_from_directory
from flask_restful import Resource, Api
from flask_cors import CORS  # Enables cross-origin requests if needed
import faiss
import json
from sentence_transformers import SentenceTransformer

# --- Flask Setup ---
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS if you need to access the API from other origins
api = Api(app)

# --- Load the Sentence Transformer model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load FAISS index and metadata ---
faiss_index_path = "courses_faiss.index"
metadata_path = "courses_metadata.json"

# Load FAISS index
index = faiss.read_index(faiss_index_path)

# Load metadata and convert keys to integers
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)
metadata = {int(k): v for k, v in metadata.items()}

# --- Define the Search Resource ---
class CourseSearch(Resource):
    def post(self):
        data = request.get_json()
        query = data.get("query", "")
        if not query:
            return jsonify({"error": "Please provide a 'query' in the request body."})
        
        # Generate an embedding for the query text
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Number of nearest neighbors to return (default is 5)
        k = data.get("k", 5)

        # Search the FAISS index for the k most similar courses
        distances, indices = index.search(query_embedding, k)
        
        # Prepare the results list using the metadata
        results = []
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), start=1):
            course_info = metadata.get(idx, {})
            results.append({
                "rank": rank,
                "course_name": course_info.get("Course Name", "N/A"),
                "course_link": course_info.get("Course Link", "N/A"),
                "course_details": course_info.get("Course Details", "N/A"),
                "course_price": course_info.get("Course Price", "N/A"),
                "number_of_lessons": course_info.get("Number of Lessons", "N/A"),
                "course_description": course_info.get("Course Description", "N/A"),
                "course_curriculum": course_info.get("Course Curriculum", "N/A"),
                "distance": float(distance)
            })
        
        return jsonify({"results": results})

# Register the /search endpoint with the API
api.add_resource(CourseSearch, '/search')

# --- Serve the static HTML page at the root URL ---
@app.route('/')
def serve_index():
    # Serve the index.html file from the "static" folder
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Run the Flask app on 0.0.0.0:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
