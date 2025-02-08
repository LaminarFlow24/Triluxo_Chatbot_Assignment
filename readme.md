# Course Chatbot

A conversational chatbot that retrieves and displays course information by leveraging web scraping, sentence embeddings, and FAISS-based vector search. The chatbot is built using Flask, LangChain, and a free Hugging Face model.

## Steps to Run

1. **Download or Clone the Repository**  
   Download the zip file or clone the Git repository to your local machine.

2. **Install Dependencies**  
   Open a command prompt (or terminal) in the project folder and run:

*Note: This process may take some time.*

3. **Run the Application**  
Execute the main Flask app by running: **python chatbot_app.py**

Once the server starts, click on the provided local port URL (e.g., [http://127.0.0.1:5000](http://127.0.0.1:5000)) in your browser.

4. **Interact with the Chatbot**  
Your application should now be running. You can interact with the chatbot through the web interface.

## Approach

1. **Data Scraping:**  
- The project scrapes not only the main webpage but also the individual course pages to extract detailed information such as "Curriculum" and "Description".

2. **Sentence Embeddings:**  
- The "all-MiniLM-L6-v2" model is used to generate sentence embeddings for the course information.

3. **Vector Store with FAISS:**  
- The generated sentence embeddings are stored using FAISS (Facebook AI Similarity Search) for efficient similarity-based retrieval.

4. **Conversational Chatbot:**  
- A Flask RESTful API is built to handle user queries.
- The API uses LangChain to combine the FAISS vector store and a free Hugging Face model (google/flan-t5-base) to power a conversational retrieval chain.
- The chatbot retrieves and displays course cards in an organized format, including details like course name (in bold and uppercase), clickable links, number of lessons, total price (computed as price × number of lessons), and actions (e.g., Book a Free Demo, Enquire Now).

## Project Structure

- **generate_embeddings.py**  
- Script to scrape course data, compute sentence embeddings, and store the FAISS index and metadata.

- **chatbot_app.py**  
- Flask application that loads the precomputed FAISS index and metadata, reconstructs the vector store, and serves the chatbot API.

- **static/index.html**  
- Frontend for the chatbot that provides an interactive UI with typing effects and organized display of course cards.

- **requirements.txt**  
- List of Python dependencies required to run the project.

## Deployment

This project can be deployed on hosting platforms such as Render, which support continuously running Flask web services. Ensure you have the appropriate configuration files (e.g., a Procfile) and environment settings if you plan to deploy.

## Notes

- **Warnings:**  
The application suppresses warnings during runtime so that only the hosted link is displayed on the terminal.
- **Dependencies:**  
Consider updating deprecated imports to use the new `langchain_community` modules as suggested in the warnings.
- **Customization:**  
You can further customize the chatbot’s behavior and UI as needed.

---

Contact me on yashasjain247@gmail.com or 8208472301 if there are any issues with running the code. 

Thank You 
