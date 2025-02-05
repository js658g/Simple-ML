"""
Uses langchain and opensource models to run Q&A on web content.

Setup:
Prepare .env file with LANGCHAIN & HF API keys:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=
HUGGINGFACEHUB_API_TOKEN=

Usage:
python rag_langchain.py

Sample output:
LoRa (Low-Rank Adaptation) and DoRA (Weight-Decomposed Low-Rank Adaptation) are both techniques used for fine-tuning language models. LoRa decomposes the weight matrix into low-rank components. DoRA, on the other hand, decomposes the weight matrix into weight components and then applies low-rank adaptation. The main difference lies in the way they decompose the weight matrix.

Sample LangSmith trace:
https://smith.langchain.com/public/c1b15afb-2b1f-4ee7-b948-8e412b3c1193/r
"""

import bs4
import dotenv
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load env vars stored in ~/.env
dotenv.load_dotenv()

# Load, chunk and index the contents of the blog.
lora_path = "https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms"
dora_path = "https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch"
loader = WebBaseLoader(
    web_paths=(lora_path, dora_path),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Specify Embedding Model
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cuda:0'})

# Use Chroma as vector DB
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Specify LLM
llm_repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_kwargs = {"temperature": 0.5, "max_length": 4096, "max_new_tokens": 1024}
llm = HuggingFaceEndpoint(repo_id=llm_repo_id, **model_kwargs)

# Chain everything together
rag_chain = (
  {"context": retriever | format_docs, "question": RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)


response = rag_chain.invoke("What is the difference between LoRA and DoRA?")
print(response)

# cleanup
vectorstore.delete_collection()
