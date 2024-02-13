from dotenv import load_dotenv
load_dotenv()  # Loads variables from the default .env file

import os
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")

from trulens_eval import Tru
tru = Tru()

from llama_index import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleWebPageReader(
    html_to_text=True
).load_data(["https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"])
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("When is the next FIFA world cup?")
print(response)

