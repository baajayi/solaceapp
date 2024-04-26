import os
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import gradio as gr


_ = load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
docs = []
loader = YoutubeTranscriptReader()
docs.extend(loader.load_data(ytlinks = ["https://www.youtube.com/watch?v=oZw67iS4rYA", 
                                        "https://www.youtube.com/watch?v=L3ynZBaWHmk", 
                                        "https://www.youtube.com/watch?v=P0dl-DwZsSQ", 
                                        "https://www.youtube.com/watch?v=4mLhA3JztB0", 
                                        "https://www.youtube.com/watch?v=q0DVDVcXFYk", 
                                        "https://www.youtube.com/watch?v=SnhGqiNvce0",
                                        "https://www.youtube.com/watch?v=CnWKIUA811s",
                                        "https://www.youtube.com/watch?v=ADPaOBzaZLo",
                                        "https://www.youtube.com/watch?v=LoSpNWFzF-k"]))

test_prompts = [
    "Tell me about Reverence",
    "Explain Salvation.",
    "What does carrying my cross entails",
   
]
context = "\n\n".join(doc.text for doc in docs)

split_text = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = split_text.split_text(context)


embed_model = OpenAIEmbeddings()
vector_index = Chroma.from_texts(texts, embed_model).as_retriever()


llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

llm_prompt_template = """You are a helpful assistant. Use the context provide to respond to the query. 
                 Let the tone of the context dictate the tone of your response. 
                 Take it step by step. Also know that the context is always about the Christian Religion, so let your answers align.
                 Please provide Bible references for your answers.\n
Question: {question} \nContext: {context} \nAnswer:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

rag_chain = (
    {"context": vector_index, "question": RunnablePassthrough() }
    | llm_prompt
    | llm
    | StrOutputParser()
)



def answer_question(question):
    answer = rag_chain.invoke(question)
    return answer

# for prompt in test_prompts:
#     response = get_retrieval_augmented_response(query_engine, prompt)
#     print(f"Prompt: {prompt}\nResponse: {response}\n")

def gradio_interface(prompt):
    response = answer_question(prompt)
    return response

interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs=gr.Textbox(),
    title="VIVE Church AI Assistant",
    description="Enter a query to get insights from an AI that leverages Youtube transcripts and OpenAI models."
)

interface.launch(share=True)

