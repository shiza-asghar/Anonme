from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf
from io import BytesIO
import librosa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import faiss
import pickle

# Initialize FastAPI
app = FastAPI()

# Initialize components
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda:0" else torch.float32

model_id = "openai/whisper-large-v3-turbo"
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)
whisper_processor = AutoProcessor.from_pretrained(model_id)

faiss_index_file = "faiss_index.index"
embeddings_file = "embeddings.pkl"

faiss_index = faiss.read_index(faiss_index_file)
with open(embeddings_file, "rb") as f:
    texts = pickle.load(f)

# Prepare the docstore and mapping
docstore = InMemoryDocstore({str(i): texts[i] for i in range(len(texts))})
index_to_docstore_id = {i: str(i) for i in range(len(texts))}

# Initialize FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS(
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embeddings.embed_query
)

groq_api_key = "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"
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.3,
    api_key=groq_api_key,
)

template = """Use the following context to quickly answer the question at the end with a tone reminiscent of Elon Musk. The response should be direct, futuristic, and visionary, consisting of no more than two succinct sentences. If the answer is unknown, state that clearly without speculation. Infuse the response with Musk's characteristic sense of humor whenever possible. 
{context}
Question: {question}
Visionary Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={'prompt': QA_CHAIN_PROMPT}
)


# Define API input schema for chatbot
class QueryInput(BaseModel):
    query: str

# Endpoint to handle chatbot queries
@app.post("/chat")
def chatbot_query(input_data: QueryInput):
    query = input_data.query
    try:
        response = qa_chain.invoke({'query': query})
        result = {
            "query": response["query"],
            "result": response["result"],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in response["source_documents"]
            ]
        }
        # Return only the answer (without source content)
        return {"result": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Endpoint to handle transcription of audio files
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Read the uploaded file into memory
        audio_data = await file.read()
        audio_file = BytesIO(audio_data)

        # Use soundfile to read the audio file content and convert it to a numpy array
        audio, sample_rate = sf.read(audio_file)

        # Resample the audio to 16000 Hz if it's not already at that rate
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        # Process the audio with Whisper
        inputs = whisper_processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs["input_features"].to(device)  # Use input_features instead of input_ids

        # Generate predictions using input_features
        predicted_ids = whisper_model.generate(input_features)

        # Decode the predicted ids to text
        transcript = whisper_processor.decode(predicted_ids[0])
        try:
            response = qa_chain.invoke({'query': transcript})
            result = {
                "query": response["query"],
                "result": response["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in response["source_documents"]
                ]
            }
            # Return only the answer (without source content)
            return {"result": result["result"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing audio query: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transcription: {str(e)}")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Elon Musk-inspired chatbot API!"}
