from datasets import load_dataset
from tqdm import tqdm

from qasper.dataset_reader import QasperReader
from qasper.models import qasper, gpt35

from qasper.evaluator import token_f1_score, get_answers_and_evidence, evaluate


import random

def instance_generator(split, reader):
    for article in split:
        for instance in reader._article_to_instances(article):
            yield instance

def evaluate(gold, predicted):
    max_answer_f1s = []
    max_evidence_f1s = []
    max_answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    num_missing_predictions = 0
    for question_id, references in gold.items():
        if question_id not in predicted:
            num_missing_predictions += 1
            max_answer_f1s.append(0.0)
            max_evidence_f1s.append(0.0)
            continue
        answer_f1s_and_types = [
            (token_f1_score(predicted[question_id]["answer"], reference["answer"]),
             reference["type"])
            for reference in gold[question_id]
        ]
        max_answer_f1, answer_type = sorted(answer_f1s_and_types, key=lambda x: x[0], reverse=True)[0]
        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type[answer_type].append(max_answer_f1)
        # evidence_f1s = [
        #     paragraph_f1_score(predicted[question_id]["evidence"], reference["evidence"])
        #     for reference in gold[question_id]
        # ]
        # max_evidence_f1s.append(max(evidence_f1s))

    mean = lambda x: sum(x) / len(x) if x else 0.0
    return {
        "Answer F1": mean(max_answer_f1s),
        "Answer F1 by type": {key: mean(value) for key, value in max_answer_f1s_by_type.items()},
        # "Evidence F1": mean(max_evidence_f1s),
        "Missing predictions": num_missing_predictions
    }


def main():

    import pdb; pdb.set_trace()

    dataset = load_dataset("allenai/qasper")
    print(dataset['validation'])
    reader = QasperReader()
    
    # randomly sample 100 instances
    random.seed(42)
    instances = list(instance_generator(dataset['validation'], reader))
    instances = random.sample(instances, 100)

    instance = instances[0]
    print(instance.keys())
    print('QUESTION WITH CONTEXT:', instance['s_question_with_context'])


    print('QUESTION:', instance['metadata']['question'])
    print('ANSWER:', instance['answer'])

    qasper_answer = qasper.predict(instance)[0]
    print(qasper_answer)

    token_f1_score(qasper_answer, instance['answer'])

    gpt35_answer = gpt35.predict(instance)
    print(gpt35_answer)

    token_f1_score(gpt35_answer, instance['answer'])



    gold_answers_and_evidence = get_answers_and_evidence(dataset['validation'])

    print(gold_answers_and_evidence.keys())

    predicted_answers_and_evidence = {}
    for instance in tqdm(instances):
        question_id = instance["metadata"]["question_id"]
        # prediction_data = json.loads(line)
        pred_answer = qasper.predict(instance)[0]

        predicted_answers_and_evidence[question_id] = {
            "answer": pred_answer,
            # "evidence": prediction_data["predicted_evidence"]
        }

    evaluation_output = evaluate(
        {k:v for k, v in gold_answers_and_evidence.items() \
            if k in predicted_answers_and_evidence}, 
        predicted_answers_and_evidence)

    len(gold_answers_and_evidence)

    evaluation_output

if __name__ == "__main__": 
    main()