Here’s the revised **README.md** with an explanation of the code and without the project structure:

---

# RAG Chatbot with Hugging Face and FAISS  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD641?style=for-the-badge&logo=huggingface&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-0A66C2?style=for-the-badge&logo=langchain&logoColor=white)

This project demonstrates a **Retrieval-Augmented Generation (RAG)** chatbot. It combines **document retrieval** with **text generation** using open-source tools like Hugging Face Transformers, FAISS, and LangChain.

---

## 📋 **How It Works**
The RAG chatbot integrates:
1. **Document Retrieval**: FAISS retrieves relevant chunks of information from the dataset based on user input.
2. **Answer Generation**: Hugging Face's `Flan-T5` generates answers based on the retrieved context.
3. **Interactive Chat**: A Streamlit interface allows users to ask questions and view answers.

---

## 🛠 **Technologies Used**
- **[Python](https://www.python.org/)**: Core programming language.
- **[Hugging Face](https://huggingface.co/)**: Generative AI models.
- **[FAISS](https://github.com/facebookresearch/faiss)**: For fast similarity searches.
- **[LangChain](https://www.langchain.com/)**: Framework for linking retrieval and generation.


---



## 🔧 **Code Explanation**
### **1. Loading the Vector Store**
The FAISS vector store is used for storing and retrieving document embeddings:
```python
@st.cache_resource
def load_vectorstore():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings)
```
- **SentenceTransformer** generates embeddings for textual data.
- **FAISS** stores these embeddings for similarity-based retrieval.

---

### **2. Loading the Hugging Face Generator**
A Hugging Face pipeline generates answers:
```python
@st.cache_resource
def load_generator():
    return pipeline("text2text-generation", model="google/flan-t5-large")
```
- **`google/flan-t5-large`**: A powerful open-source generative model for question answering.

---

### **3. Retrieving Documents**
This function retrieves the top `k` relevant documents:
```python
def retrieve_documents(query, vectorstore, top_k=3):
    retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    combined_content = " ".join([doc.page_content for doc in retrieved_docs])
    return combined_content, retrieved_docs
```
- **Similarity Search**: Finds the most relevant chunks of data for the query.
- **Combined Content**: Concatenates the retrieved chunks for use as context.

---

### **4. Generating Answers**
The retrieved context is passed to the Hugging Face generator to produce a response:
```python
def generate_answer(prompt, generator):
    result = generator(prompt, max_length=200, num_return_sequences=1)
    return result[0]["generated_text"]
```
- **Prompt**: Combines the retrieved context and user query.
- **Generated Text**: The model returns a concise answer based on the prompt.

---

### **5. Streamlit Interface**
A user-friendly web interface built with Streamlit:
```python
st.title("RAG Chatbot")
st.write("Ask any question based on the knowledge base!")

query = st.text_input("Enter your question:")
if query:
    vectorstore = load_vectorstore()
    generator = load_generator()
    context, source_docs = retrieve_documents(query, vectorstore)
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"
    answer = generate_answer(prompt, generator)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Source Documents")
    for doc in source_docs:
        st.write(f"- **{doc.metadata['question']}**: {doc.page_content}")
```
- **`st.text_input`**: Captures user questions.
- **Answer Generation**: Combines retrieval and generation pipelines to produce an answer.
- **Source Documents**: Displays the documents used for generating the answer.

---

## ✨ **Key Features**
- **Open-Source Models**: No reliance on paid APIs like OpenAI.
- **Custom Knowledge Base**: Use your own dataset for specific domains.
- **PDF Logging**: Save conversations to a PDF file for record-keeping.

---

## 📖 **Usage Example**
### Interaction Example:
```plaintext
User: What is Artificial Intelligence?
Chatbot: Artificial Intelligence refers to systems capable of simulating human intelligence.
```

---

## 👩‍💻 **Future Enhancements**
1. Add multi-language support.
2. Integrate voice input/output.
3. Enhance the retrieval mechanism with more advanced embeddings.

---

## 📝 **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Let me know if you'd like further enhancements or customizations!
