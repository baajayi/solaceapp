import threading
from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import os
import json
import concurrent.futures
import numpy as np
import tiktoken
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv, find_dotenv
import gradio as gr

app = Flask(__name__)

_ = load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
docs = []
links = [
    "https://www.youtube.com/watch?v=jMPP8RH3-MQ",
    "https://www.youtube.com/watch?v=8DIveZcmcnA",
    "https://www.youtube.com/watch?v=CnWKIUA811s",
    "https://www.youtube.com/watch?v=ADPaOBzaZLo",
    "https://www.youtube.com/watch?v=oZw67iS4rYA",
    "https://www.youtube.com/watch?v=y7UN3MY-LIs",
    "https://www.youtube.com/watch?v=L3ynZBaWHmk",
    "https://www.youtube.com/watch?v=gv6WCCWZC14",
    "https://www.youtube.com/watch?v=LoSpNWFzF-k",
    "https://www.youtube.com/watch?v=w_osZN17DLw",
    "https://www.youtube.com/watch?v=r6OqRIQzDA8",
    "https://www.youtube.com/watch?v=q0DVDVcXFYk",
    "https://www.youtube.com/watch?v=unZf8wZZXkw",
    "https://www.youtube.com/watch?v=co2KJ9zKNbA",
    "https://www.youtube.com/watch?v=vqf2mnf8PUg",
    "https://www.youtube.com/watch?v=KN5kekYwmUc",
    "https://www.youtube.com/watch?v=-HtFRj8jnSU",
    "https://www.youtube.com/watch?v=BtLkQmhg2RU",
    "https://www.youtube.com/watch?v=NRffr8LanRU",
    "https://www.youtube.com/watch?v=6SkY8pG6lHk"
]

youtube_links = [link.split("=")[1] for link in links]

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([t['text'] for t in transcript])
    return text

def split_text(text, max_tokens, encoding_name="cl100k_base"):
    tokenizer = tiktoken.get_encoding(encoding_name)
    tokens = tokenizer.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
    
    return chunks

def get_embeddings_for_text_chunks(text_chunks, model="text-embedding-ada-002"):
    embeddings = []
    for chunk in text_chunks:
        response = client.embeddings.create(input=chunk, model=model)
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    return embeddings

def save_embeddings_to_json(embeddings, filename):
    with open(filename, 'w') as f:
        json.dump(embeddings, f)
    print(f"Embeddings saved to {filename}")

def save_embeddings_to_npy(embeddings, filename):
    np.save(filename, embeddings)
    print(f"Embeddings saved to {filename}")

def load_embeddings_from_json(filename):
    with open(filename, 'r') as f:
        embeddings = json.load(f)
    return embeddings

def load_embeddings_from_npy(filename):
    return np.load(filename, allow_pickle=True).tolist()

def is_valid_embedding(embedding):
    return np.all(np.isfinite(embedding))

embeddings_file = 'vivembeddings.npy'
document_texts_file = 'vivdocument_texts.json'

if os.path.exists(embeddings_file) and os.path.exists(document_texts_file):
    all_embeddings = load_embeddings_from_npy(embeddings_file)
    with open(document_texts_file, 'r') as f:
        document_texts = json.load(f)
else:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_transcripts = executor.map(get_transcript, youtube_links)
        docs.extend(future_transcripts)
    
    all_embeddings = []
    document_texts = []
    for doc in docs:
        text_chunks = split_text(doc, max_tokens=8192)
        embeddings = get_embeddings_for_text_chunks(text_chunks)
        all_embeddings.extend(embeddings)
        document_texts.extend(text_chunks)
    
    save_embeddings_to_npy(all_embeddings, embeddings_file)
    with open(document_texts_file, 'w') as f:
        json.dump(document_texts, f)

valid_embeddings = []
valid_texts = []
for embedding, text in zip(all_embeddings, document_texts):
    if is_valid_embedding(embedding):
        valid_embeddings.append(embedding)
        valid_texts.append(text)

def find_similar_documents(query_embedding, embeddings, texts, top_k=5):
    similarities = cosine_similarity([query_embedding], embeddings)
    top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
    return [(texts[i], similarities[0][i]) for i in top_k_indices]

def get_retrieval_augmented_response(prompt, model="text-embedding-ada-002"):
    query_embedding_response = client.embeddings.create(input=prompt, model=model)
    query_embedding = query_embedding_response.data[0].embedding
    similar_docs = find_similar_documents(query_embedding, valid_embeddings, valid_texts)

    max_context_length = 8192
    context = ""
    total_length = 0
    tokenizer = tiktoken.get_encoding("cl100k_base")
    for doc, _ in similar_docs:
        doc_length = len(tokenizer.encode(doc))
        if total_length + doc_length <= max_context_length:
            context += doc + " "
            total_length += doc_length
        else:
            break

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a spiritual assistant. You have the context of sermons by Pastor Adam Smallcombe of the VIVE Church. 
                 Your role is to answer questions in a personal, warm, and engaging manner that reflects the style and tone of Pastor Adam Smallcombe, 
                 but without repeating the sermons. Directly address the person asking the question, offering guidance and scriptural references as appropriate.
                 Ensure the response is conversational and focused on the individual's query, providing support and encouragement in the spirit of the sermons. 
                 Use the following context to help shape your response:"""},
                {"role": "user", "content": context.strip()},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def gradio_interface(query):
    return get_retrieval_augmented_response(query)

gradio_app = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your Question here..."),
    outputs=gr.TextArea(),
    title="VIVE AI Assistant",
    description="Ask questions and get responses inspired by VIVE Church."
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query', '')
    response = get_retrieval_augmented_response(query)
    return jsonify({"response": response})

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=True)

def run_gradio():
    gradio_app.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    gradio_thread = threading.Thread(target=run_gradio)

    flask_thread.start()
    gradio_thread.start()

    flask_thread.join()
    gradio_thread.join()
