
def transpose_dict(d):
    arr = []
    keys = d.keys()
    for question_answer in zip(*[d[key] for key in keys]):
        arr.append(dict(zip(keys, question_answer)))
    return arr