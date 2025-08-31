# Farm_Bot 🌱🤖

Farm_Bot is an AI-powered **multilingual (English & Hindi) assistant** designed to support farmers and agriculture-related queries. It integrates modern AI techniques like **RAG (Retrieval-Augmented Generation)**, **Hybrid System Architecture**, and **CNN-based models for disease/crop (DC) detection** to deliver accurate and real-time insights.

---

## 🚀 Features
- **Multilingual Support:** 22 indian Languages
- **Hybrid RAG System:** Combines knowledge retrieval with generative AI for precise answers
- **CNN-based Plant Disease Detector:** Crop/disease detection using deep learning (TensorFlow)
- **Resource Allocation:** TensorFlow GPU assigned for CNN model, CPU assigned for other processing to avoid conflicts
- **Farmer-Friendly UI:** Simple and easy to interact with
- **Scalable Design:** Future-ready for deployment on cloud platforms

---

## 🛠️ Tech Stack
- **Frameworks & Libraries:** Python, TensorFlow, Transformers, LangChain
- **Modeling:** CNN for disease detection, RAG for chatbot
- **Tools:** FAISS (for vector search in RAG)
- **Backend:** Groq API
- **Frontend:** Streamlit / Web UI
- **Database:** Custom Agricultural Dataset + Knowledge Base

---

## 📂 Project Workflow
1. **Data Collection - (Custom Agricultural Datasets)**
2. **Preprocessing & Feature Engineering**
3. **CNN Model Training (Disease/Crop Detection)**
4. **RAG Pipeline Integration for Chatbot**

---

## ⚡ Current Status
- ✅ Chatbot is functional locally  
- ✅ RAG and CNN models trained & tested  
- ⚠️ **Deployment Pending**: Due to **memory shortage issues**, the project has not yet been deployed to cloud platforms like Streamlit cloud/Render.  
- Future plan: Optimize resource usage and deploy on scalable infrastructure  

---

## 👨‍💻 Team
- **Ansari Sumaiya**  
- **Om Singh**
- **Zafars Ansari**
- **Rishika Banda**

---

## 🎓 Guidance
Special thanks to **Professor Shobhit Singh** for valuable guidance and mentorship throughout this project. 🙏

---

## 📌 Notes
- Ensure **TensorFlow GPU** is strictly assigned to the CNN model and **CPU** is assigned to the chatbot RAG pipeline to avoid resource conflicts.
- Additional optimization for deployment is planned in the next iteration.

---

## 🔮 Future Improvements
- Deploy the chatbot on cloud (AWS / GCP / Vercel / Render)  
- Expand dataset for better crop/disease predictions  
- Introduce voice-based interaction for farmers  

---
