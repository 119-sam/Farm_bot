import os
import json
import faiss
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
import io
import base64

app = Flask(__name__)

# Initialize components
def initialize_components():
    global qa_chain, llm, model, labels, embedding_model
    
    # Get API key
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load FAISS vectorstore
    try:
        index = faiss.read_index("index.faiss")
        with open("metadata.json", "r") as f:
            metadata = json.load(f)
        docstore = {k: Document(**v) for k, v in metadata["docstore"].items()}
        vectordb = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=metadata["index_to_docstore_id"]
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        print("FAISS vectorstore loaded successfully")
    except Exception as e:
        print(f"Error loading FAISS: {e}")
        retriever = None
    
    # Initialize LLM
    if GROQ_API_KEY:
        try:
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="mixtral-8x7b-32768",
                temperature=0.1
            )
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            llm = None
    else:
        llm = None
        print("GROQ_API_KEY not set")
    
    # Create QA chain
    if retriever and llm:
        try:
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )
            print("QA chain created successfully")
        except Exception as e:
            print(f"Error creating QA chain: {e}")
            qa_chain = None
    else:
        qa_chain = None
        print("QA chain not available due to missing components")
    
    # Load disease detection model
    try:
        model = tf.keras.models.load_model("Plant_Disease_CNN_model.h5")
        with open("class_indices.json", "r") as f:
            class_indices = json.load(f)
        labels = {v: k for k, v in class_indices.items()}
        print("Disease detection model loaded successfully")
    except Exception as e:
        print(f"Error loading disease model: {e}")
        model, labels = None, None

# Initialize components when app starts
initialize_components()

# Chat history
chat_history = []

def call_llm_direct(user_msg):
    if not llm:
        return "LLM not available. Please check your API key."
    
    system_prompt = "You are an agronomy assistant. Be concise and helpful about crops, soil, irrigation, pests, and fertilizers."
    
    history_txt = ""
    if chat_history:
        lines = []
        for u, a in chat_history[-6:]:
            lines.append(f"User: {u}\nAssistant: {a}")
        history_txt = "\n\n".join(lines)

    prompt = f"{system_prompt}\n\n"
    if history_txt:
        prompt += f"Conversation so far:\n{history_txt}\n\n"
    prompt += f"User: {user_msg}\nAssistant:"

    try:
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        print(f"LLM error: {e}")
        return "Sorry, I'm currently unable to answer your question."

def preprocess_image(img_file):
    img = Image.open(io.BytesIO(img_file)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_msg = request.json.get('message', '')
        
        if not user_msg:
            return jsonify({'error': 'No message provided'})
        
        if qa_chain:
            result = qa_chain({"question": user_msg, "chat_history": chat_history})
            reply = result["answer"]
        else:
            reply = call_llm_direct(user_msg)
        
        chat_history.append((user_msg, reply))
        
        return jsonify({
            'response': reply,
            'history': chat_history[-10:]  # Return last 10 messages
        })
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'error': str(e)})

@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if model is None:
            return jsonify({'error': 'Disease detection model not available'})
        
        img_data = file.read()
        img_array = preprocess_image(img_data)
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        class_label = labels[class_idx]
        confidence = float(np.max(prediction)) * 100
        
        # Convert image to base64 for display
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        return jsonify({
            'disease': class_label,
            'confidence': confidence,
            'image': f"data:image/jpeg;base64,{img_base64}"
        })
    except Exception as e:
        print(f"Disease detection error: {e}")
        return jsonify({'error': str(e)})

@app.route('/get_history', methods=['GET'])
def get_history():
    return jsonify({'history': chat_history[-10:]})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'llm_available': llm is not None,
        'model_available': model is not None,
        'qa_chain_available': qa_chain is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

