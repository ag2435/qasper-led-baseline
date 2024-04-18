from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained('allenai/led-base-16384')
qasper_led = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384")

def predict(instance):
    tokens = instance['question_with_context']
#     global_attention_mask = torch.tensor([list(instance['global_attention_mask'].array)])
    global_attention_mask = instance['global_attention_mask']
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([token_ids])
    attention_mask = torch.tensor([[True] * len(token_ids)])
    # if not args.cpu:
    #     global_attention_mask = global_attention_mask.cuda()
    #     input_ids = input_ids.cuda()
    #     attention_mask = attention_mask.cuda()
    generation_output = qasper_led.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
        do_sample=True,
        num_return_sequences=1, #args.samples,
        output_scores=True,
        return_dict_in_generate=True
    )
    # return generation_output

    output_sequences = generation_output.sequences.tolist()

    answers = []
    for sequence in output_sequences:
        answers.append(tokenizer.decode(sequence, skip_special_tokens=True))

    return answers
