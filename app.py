import os
import streamlit as st
port = int(os.environ.get("PORT", 8501))

import json
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain


st.set_page_config(page_title="AgriBot 🌾", layout="wide")


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 1em;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 8px;
        background: #f1f8e9;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .stChatMessage.user {
        background: #e3f2fd;
        text-align: right;
    }
    .stChatMessage.assistant {
        background: #fffde7;
        text-align: left;
    }
    .stChatInput textarea {
        border-radius: 8px;
        border: 1px solid #2e7d32;
        padding: 8px;
        font-size: 1rem;
    }
    </style>
    <div class="main-title">🌾 AgriBot — Farm Assistant</div>
    <div class="subtitle">Chat about crops, soil, irrigation & pests • <b>Image disease detector</b></div>
    """,
    unsafe_allow_html=True
)


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")



def load_faiss_vectorstore(embedding_model):
    index_path = "index.faiss"
    metadata_path = "metadata.json"

    
    
    index = faiss.read_index(index_path)

    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    
    docstore = {k: Document(**v) for k, v in metadata["docstore"].items()}

    
    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=metadata["index_to_docstore_id"]
    )
    return vectorstore



embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = load_faiss_vectorstore(embedding_model)


retriever = vectordb.as_retriever(search_kwargs={"k": 3})




llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b"   
)


qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def call_llm_direct(llm, user_msg, chat_history=None):
    
    system_prompt = (
        "You are an agronomy assistant. Be concise and helpful about crops, soil, irrigation, pests, and fertilizers."
    )
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
       if hasattr(resp, "content"):
         return resp.content
       return str(resp)
    except Exception as e:
      st.error(f"LLM error: {e}")
      return "Sorry, I'm currently unable to answer your question."





left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("💬 Chatbot")  
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! Ask me about crop care, soil, irrigation, pests, or fertilizers."}
        ]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    
    chat_container = st.container()

    
    user_msg = st.chat_input("Type your question...")

    
    with chat_container:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

    
    if user_msg:
        
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with chat_container.chat_message("user"):
            st.write(user_msg)

        
            try:
                result = qa_chain({"question": user_msg, "chat_history": st.session_state.chat_history})
                reply = result["answer"]
            except Exception as e:
                #st.warning(f"Retrieval error (ignored): {e}")
                # Fallback: directly call LLM without retrieval
                reply = call_llm_direct(llm, user_msg, chat_history=st.session_state.chat_history)

            
            st.session_state.chat_history.append((user_msg, reply))

        
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with chat_container.chat_message("assistant"):
            st.write(reply)



import numpy as np
from PIL import Image
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json


def preprocess_image(img_file):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array


@st.cache_resource
def load_cnn_model_and_labels():
    model = tf.keras.models.load_model("Plant_Disease_CNN_model.h5")
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

model, labels = load_cnn_model_and_labels()

with right:
    
    st.markdown(
    """
    <div style="margin-top:-30px;">
        <h3 style='color:#c62828;'>🧪 Plant Disease Detector</h3>
    </div>
    """,
    unsafe_allow_html=True
)

    uploaded_file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        class_label = labels[class_idx]
        confidence = float(np.max(prediction)) * 100

        st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)
        st.success(f"**Predicted Disease:** `{class_label}`")
        st.info(f"**Confidence:** `{confidence:.2f}%`")

        
        disease_query = f"I have detected '{class_label}' disease in my plant. What should I do?"
        if "disease_sent" not in st.session_state or st.session_state.disease_sent != class_label:
            st.session_state.disease_sent = class_label
            st.session_state.messages.append({"role": "user", "content": disease_query})
            with left:
                with st.chat_message("user"):
                    st.write(disease_query)
                with st.spinner("Thinking..."):
                    try:
                        result = qa_chain({"question": disease_query, "chat_history": st.session_state.chat_history})
                        reply = result["answer"]
                    except Exception as e:
                        reply = call_llm_direct(llm, disease_query, chat_history=st.session_state.chat_history)
                    st.session_state.chat_history.append((disease_query, reply))
                st.session_state.messages.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.write(reply)
    else:
        st.info("Upload a leaf image to get a diagnosis.")


if __name__ == "__main__":
    st.run("app.py", server_port=port, server_address="0.0.0.0")



