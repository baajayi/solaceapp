from llama_index import WikipediaReader
import os
from llama_index.vector_stores import MilvusVectorStore
from llama_index.llms import LangChainLLM, OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from llama_index.storage.storage_context import StorageContext
from llama_index import ServiceContext, VectorStoreIndex
from tenacity import retry, stop_after_attempt, wait_exponential
from trulens_eval import Tru, feedback
from trulens_eval.feedback import Groundedness
from trulens_eval import Feedback
from trulens_eval import TruLlama
import numpy as np
import itertools
import nltk
nltk.download('punkt')

os.environ["OPENAI_API_KEY"] = "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"



cities = [
    "Los Angeles", "Houston", "Honolulu", "Tucson", "Mexico City", 
    "Cincinatti", "Chicago"
]

wiki_docs = []
for city in cities:
    try:
        doc = WikipediaReader().load_data(pages=[city])
        wiki_docs.extend(doc)
    except Exception as e:
        print(f"Error loading page for city {city}: {e}")


test_prompts = [
    "What's the best national park near Honolulu",
    "What are some famous universities in Tucson?",
    "What bodies of water are near Chicago?",
    "What is the name of Chicago's central business district?",
    "What are the two most famous universities in Los Angeles?",
    "What are some famous festivals in Mexico City?",
    "What are some famous festivals in Los Angeles?",
    "What professional sports teams are located in Los Angeles",
    "How do you classify Houston's climate?",
    "What landmarks should I know about in Cincinatti"
]

############################################################################
# TODO: Initialize MilvusVectorStore with a vector index of your choice.   #
# Note: Some vector indexes may not require search_params.                 #
############################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

vector_store = MilvusVectorStore(
    index_params={
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist":1024},
    },
    search_params={
        "metric_type": "L2",
        "params": {"nprobe": 10}
    },
    overwrite=True
)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
############################################################################
#                             END OF YOUR CODE                             #



# llm = LangChainLLM(llm=Cohere(model="command"))
llm = OpenAI(model="gpt-3.5-turbo",api_key="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")

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
############################################################################
#                             END OF YOUR CODE                             #
############################################################################




############################################################################
# TODO: Initialize a model embedding of your choice.                       #
############################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# embed_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
# )
embed_model = OpenAIEmbeddings(openai_api_key="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")

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
############################################################################
#                             END OF YOUR CODE                             #
############################################################################


storage_context = StorageContext.from_defaults(vector_store = vector_store)
service_context = ServiceContext.from_defaults(embed_model = embed_model, llm = llm)
index = VectorStoreIndex.from_documents(wiki_docs,
            service_context=service_context,
            storage_context=storage_context)

query_engine = index.as_query_engine(top_k = 5)

# adds exponential backoff for LLM queries
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_query_engine(prompt):
        return query_engine.query(prompt)

for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        print(f"Response: {call_query_engine(prompt)}\n")


# init trulens
tru = Tru()

# Initialize OpenAI-based feedback function collection class
openai_gpt35 = feedback.OpenAI(model_engine="gpt-3.5-turbo")

# Initialize groundedness class for the groundedness metric
grounded = Groundedness(groundedness_provider=openai_gpt35)


f_answer_relevance = Feedback(openai_gpt35.relevance_with_cot_reasons, name = "Answer Relevance").on_input_output()

f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness").on(
    TruLlama.select_source_nodes().node.text # this line grabs the context that was supplied with the query
).on_output().aggregate(grounded.grounded_statements_aggregator)

############################################################################
# TODO: Initialize f_context_relevance to measure the context relevance    #
# between question and each context chunk.                                 #
############################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

f_context_relevance = Feedback(openai_gpt35.relevance_with_cot_reasons, name = "Context Relevance").on_input().on(
    TruLlama.select_source_nodes().node.text).aggregate(np.mean) 

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
############################################################################
#                             END OF YOUR CODE                             #
############################################################################



############################################################################
# TODO: Try and evaluate different RAG configurations.                     #
############################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

index_params = [
    {
        "index_param": {
            "metric_type": "L2",
            "index_type": "IVF_PQ",
            "params": {"nlist":1024, "m":8},
        },
        "search_param": {
            "metric_type": "L2",
            "params": {"nprobe": 32}
        }
     },
    {
        "index_param": {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist":1024},
        },
        "search_param": {
            "metric_type": "L2",
            "params": {"nprobe": 8}
        },
    },
    {
        "index_param": {
            "metric_type": "L2",
            "index_type": "IVF_PQ",
            "params": {"nlist":1024, "m":8},
        },
        "search_param": {
            "metric_type": "L2",
            "params": {"nprobe": 8}
        },
    },
    {
        "index_param": {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist":1024},
        },
        "search_param": {
            "metric_type": "L2",
            "params": {"nprobe": 32}
        },
    },
]
embed_models = [
    OpenAIEmbeddings(openai_api_key="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")
]
context_chunk_sizes = [200, 500]  # feel free to try others

top_ks = [1,3]

for index_param, embed_model, top_k, chunk_size in itertools.product(
    index_params, embed_models, top_ks, context_chunk_sizes
    ):
    # print(index_param, embed_model.model, top_k, chunk_size)
    embed_model_name = embed_model.model
    vector_store = MilvusVectorStore(index_params=index_param["index_param"], search_params=index_param["search_param"])
    llm = OpenAI(model="gpt-3.5-turbo",api_key="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")
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    service_context = ServiceContext.from_defaults(embed_model = embed_model, llm = llm, chunk_size=chunk_size)
    index = VectorStoreIndex.from_documents(wiki_docs,
            service_context=service_context,
            storage_context=storage_context)
    
    query_engine = index.as_query_engine(similarity_top_k = top_k)
    
    # Initialize a TruLlama wrapper to connect evaluation metrics with the query engine
    tru_query_engine = TruLlama(query_engine,
                    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
                    metadata={
                        'index_param':index_param["index_param"],
                        'embed_model':embed_model_name,
                        'top_k':top_k,
                        'chunk_size':chunk_size
                        })
    
    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_tru_query_engine(prompt):
        # we now send the prompt through the TruLlama-wrapped query engine
        return tru_query_engine.query(prompt)
    for prompt in test_prompts:
        call_tru_query_engine(prompt)
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
############################################################################
#                             END OF YOUR CODE                             #
############################################################################
        
tru.run_dashboard()