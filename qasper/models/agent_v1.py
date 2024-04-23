"""
Wrapper for arXiv agent v1
"""

from typing import Dict

from arxiv_agent.graphs.test_v1 import graph
from arxiv_agent.nodes.base import parser
from langchain_core.messages import HumanMessage

from ..utils import print_wrap

def predict(instance: Dict, verbose=0) -> str:
    arxiv_id = instance['metadata']['article_id']
    question = instance['metadata']['question']
    
    # construct query
    query = f"In the paper with arxiv identifier {arxiv_id}, {question}"

    if verbose: # stream the output
        # output = []
        events = graph.stream(
            [HumanMessage(content=query)]
        )
        for i, step in enumerate(events):
            node, output = next(iter(step.items()))
            print(f"## {i+1}. {node}")
            print_wrap(str(output))
            print("---")
            # output.append(out)

    else:
        output = graph.invoke([HumanMessage(content=query)])

    # print('34>', output[-1])
    last_out = parser.invoke(output[-1])
    # print('36>', last_out)

    # get the argument from the final_answer tool
    if last_out[0]['args']['Actions'][0]['tool'] == 'final_answer':
        pred_answer = last_out[0]['args']['Actions'][0]['argument']
    else:
        pred_answer = "Sorry, I could not find an answer to your question."

    return pred_answer