import json
import argparse
import numpy as np
from qasper.metrics.squad_em_and_f1 import SquadEmAndF1

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
    samples_data = []
    base_count = 0 # number of total questions
    sample_count = 0 # number of questions with at least 3 references

    for paper_info in data.values():
        for qa_info in paper_info["qas"]:
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
                samples_data.append(
                    {
                        "question_id": qa_info["question_id"],
                        "answers": refs,
                        "normalized_answer_log_probs": [0.0] * len(refs)
                    }
                )
                
                sample_count += 1
            
            base_count += 1
    print(f"Total questions: {base_count}")
    print(f"Questions with at least 3 references: {sample_count}")
    print(f"Percentage of questions with at least 3 references: {sample_count / base_count * 100}%")

    # samples_data = [json.loads(line) for line in open(args.samples)]
    # print(f"Read {len(samples_data)} predictions")
    oracle_f1_ranks = []
    oracle_f1s = []
    model_f1s = []
    log_file = open(args.log, "w") if args.log else None
    for prediction_info in samples_data:
        references = answers[prediction_info["question_id"]]
        predictions = prediction_info["answers"]
        scores = prediction_info["normalized_answer_log_probs"]
        sorted_predictions = [y[1] for y in sorted(zip(scores, predictions), key=lambda x: x[0], reverse=True)]
        # print(sorted_predictions)
        f1s = [np.mean([get_f1(pred, reference) for j, reference in enumerate(references) if i!=j]) for i, pred in enumerate(sorted_predictions)]
        oracle_f1 = max(f1s)
        # print(oracle_f1)
        model_f1s.append(f1s[0])
        oracle_f1s.append(oracle_f1)
        for i, f1 in enumerate(f1s):
            if f1 == oracle_f1:
                oracle_f1_ranks.append(i+1)
                break
        if log_file:
            print(
                    json.dumps(
                        {
                            "question_id": prediction_info["question_id"],
                            "question": questions[prediction_info["question_id"]],
                            "references": references,
                            "best_model_answer": sorted_predictions[0],
                            "oracle_answer": sorted_predictions[i],
                            "oracle_answer_rank": i+1
                        }
                    ),
                    file=log_file
            )

    average = lambda l: sum(l) / len(l)
    print(f"Average oracle F1 rank: {average(oracle_f1_ranks)}")
    print(f"Average model F1: {average(model_f1s)}")
    print(f"Average oracle F1: {average(oracle_f1s)}")


if __name__ == "__main__":
    main()
