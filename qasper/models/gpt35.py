"""
LangChain wrapper for GPT-3.5-turbo zero-shot
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tiktoken

MODEL_NAME = 'gpt-3.5-turbo'

# 
# OpenAI utils
# 
encoding = tiktoken.encoding_for_model(MODEL_NAME)
MAX_TOKENS = 16300 # use something slighly less than 16385

def truncate_string(string: str) -> str:
    """
    Truncate string to max tokens for GPT-3.5-turbo.
    Otherwise, we get a 400 error from OpenAI.

    Ref: 
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    # encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(string)
    
    num_tokens = len(tokens)
    if num_tokens > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]

    return encoding.decode(tokens)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

gpt35 = ChatOpenAI(
    model=MODEL_NAME
)

output_parser = StrOutputParser()

chain = prompt | gpt35 | output_parser

def predict(instance: str):
    query = instance['s_question_with_context']
    return chain.invoke({"input": truncate_string(query)})
