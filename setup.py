from dotenv import load_dotenv
load_dotenv()  # Loads variables from the default .env file
import nltk

import os
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")

from trulens_eval import Tru
tru = Tru()

from llama_index import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleWebPageReader(
    html_to_text=True
).load_data(["https://www.churchofjesuschrist.org/study/general-conference/2021/10?lang=eng"])
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("What did Presiden Russell M. Nelson speak about in the October 2022 General Conference?")
print(response)

import numpy as np

# Initialize provider class
from trulens_eval.feedback.provider.openai import OpenAI
openai = OpenAI()

# select context to be used in feedback. the location of context is app specific.
from trulens_eval.app import App
context = App.select_context(query_engine)

# imports for feedback
from trulens_eval import Feedback

# Define a groundedness feedback function
from trulens_eval.feedback import Groundedness
grounded = Groundedness(groundedness_provider=OpenAI())
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(context.collect()) # collect context chunks into a list
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = (
    Feedback(openai.qs_relevance)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)
from trulens_eval import TruLlama
tru_query_engine_recorder = TruLlama(query_engine,
    app_id='LlamaIndex_App1',
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])

# or as context manager
with tru_query_engine_recorder as recording:
    query_engine.query("What did President Russell M. Nelson speak about in the October 2022 General Conference?")
from trulens_eval import TruLlama
tru_query_engine_recorder = TruLlama(query_engine,
    app_id='LlamaIndex_App1',
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])

# The record of the app invocation can be retrieved from the `recording`:

rec = recording.get() # use .get if only one record
# recs = recording.records # use .records if multiple

# print(rec)

# The results of the feedback functions can be rertireved from the record. These
# are `Future` instances (see `concurrent.futures`). You can use `as_completed`
# to wait until they have finished evaluating.

from trulens_eval.schema import FeedbackResult

from concurrent.futures import as_completed

for feedback_future in  as_completed(rec.feedback_results):
    feedback, feedback_result = feedback_future.result()
    
    feedback: Feedback
    feedbac_result: FeedbackResult

    print(feedback.name, feedback_result.result)

records, feedback = tru.get_records_and_feedback(app_ids=["LlamaIndex_App1"])

records.head()

tru.get_leaderboard(app_ids=["LlamaIndex_App1"])

tru.run_dashboard() # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed