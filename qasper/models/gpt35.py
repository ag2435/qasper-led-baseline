"""
to run:
> chain.invoke({"input": "how can langsmith help with testing?"})
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

gpt35 = ChatOpenAI(
    model='gpt-3.5-turbo'
)

output_parser = StrOutputParser()

chain = prompt | gpt35 | output_parser

def predict(instance: str):
    query = instance['s_question_with_context']
    return chain.invoke({"input": query})