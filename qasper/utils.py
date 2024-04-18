import textwrap

# 
# Print formatting functions
# 
WRAPPER = textwrap.TextWrapper(width=80)

def print_wrap(output):
    """
    Print the output with text wrapping (80 characters per line).
    """
    # preserve paragraph structure
    paragraphs = output.split("\n")

    for paragraph in paragraphs:
        word_list = WRAPPER.wrap(text=paragraph)
        for element in word_list: 
            print(element)
        print()

# 
# Dataset manipulation functions
#
def transpose_dict(d):
    arr = []
    keys = d.keys()
    for question_answer in zip(*[d[key] for key in keys]):
        arr.append(dict(zip(keys, question_answer)))
    return arr