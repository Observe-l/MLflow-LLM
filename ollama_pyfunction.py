import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.types.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatChoice,
)
from mlflow.models import set_model
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class OllamaPyfunc(PythonModel):
    def __init__(self):
        self.model_name = None
        self.client = None

    def load_context(self, context):
        # self.model_name = context.artifacts["model_name"]
        self.llm = OllamaLLM(model="gemma3:12b")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            context.artifacts["persist_directory"],
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        self.retrievalQA = RetrievalQA.from_llm(llm=self.llm, retriever=vectorstore.as_retriever())
        # self.client = ollama.Client()

    def predict(self, context, model_input, params=None):
        messages = model_input.get("query", [])

        response = self.retrievalQA.invoke(messages)
        return response


set_model(OllamaPyfunc())