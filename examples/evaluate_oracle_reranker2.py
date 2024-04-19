import json
import argparse
import numpy as np
from qasper.metrics.squad_em_and_f1 import SquadEmAndF1
from itertools import combinations

metric = SquadEmAndF1()
f1_hash = {}
def get_f1(pred, ans):
    pred = pred.strip()
    ans = ans.strip()
    if (pred, ans) in f1_hash:
        return f1_hash[(pred, ans)]
    if (ans, pred) in f1_hash:
        return f1_hash[(ans, pred)]
    metric(pred, [ans])
    _, f1 = metric.get_metric(True)
    f1_hash[(pred, ans)] = f1
    return f1


def get_references(answers_info):
    references = []
    for answer_info in answers_info:
        answer = answer_info["answer"]
        if answer["unanswerable"]:
            references.append("Unanswerable")
        elif answer["extractive_spans"]:
            references.append(", ".join(answer["extractive_spans"]))
        elif answer["free_form_answer"]:
            references.append(answer["free_form_answer"])
        else:
            references.append("Yes" if answer["yes_no"] else "No")
    return references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    # parser.add_argument("--samples", type=str)
    parser.add_argument("--log", type=str)
    args = parser.parse_args()

    data = json.load(open(args.data))
    answers = {}
    questions = {}
    total = 0 # number of total questions
    atleast3 = 0 # number of questions with at least 3 references

    f1_list = []
    for paper_info in data.values():
        for qa_info in paper_info["qas"]:
            # print(qa_info.keys())
            # print(qa_info['search_query'])

            refs = get_references(qa_info["answers"])
            answers[qa_info["question_id"]] = refs
            questions[qa_info["question_id"]] = qa_info["question"]

            # Albert: this didn't seem to be implemented in the original code
            # According to QASPER paper (Sec. 4.1):
            # we consider a subset of the test set containing questions with 
            # at least three references (40% of the test set), evaluate each 
            # reference against the remaining, and compute an average over all
            #  such combinations
            if len(refs) >= 3:
                # samples_data.append(
                #     {
                #         "question_id": qa_info["question_id"],
                #         "answers": refs,
                #         "normalized_answer_log_probs": [0.0] * len(refs)
                #     }
                # )
                f1s = []
                # get all pairwise combinations of references
                for ref1, ref2 in combinations(refs, 2):
                    # your code here
                    f1s.append(get_f1(ref1, ref2))
                    f1_list.append(get_f1(ref1, ref2))
                # f1_list.append(np.mean(f1s))
                atleast3 += 1
            
            total += 1
    print(f"Total questions: {total}")
    print(f"Questions with at least 3 references: {atleast3}")
    print(f"Percentage of questions with at least 3 references: {atleast3 / total * 100:.2f}%")
    print(f"Average F1: {np.mean(f1_list):.2f}")

if __name__ == "__main__":
    main()
