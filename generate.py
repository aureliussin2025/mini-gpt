import torch
from tokeniser import tokenise, encode

#actual bot function
def generate(start_words, total_num_of_words, model, word_id_dict, id_word_dict, temperature):

    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")
    
    output = [start_words]

    tokenised_start_words = tokenise(start_words)

    model.eval()

    with torch.no_grad():
    
        for i in range(total_num_of_words):

            current_words = torch.tensor([encode(tokenised_start_words, word_id_dict)])

            logits = model(current_words)

            probabilities = torch.softmax(logits/temperature, dim=-1)

            next_word_id = torch.multinomial(probabilities, num_samples=1)

            next_word = id_word_dict[next_word_id.item()]

            output.append(next_word)

            tokenised_start_words = tokenised_start_words[1:] + [next_word]

    return " ".join(output)


# """
# WIP
# """
# #generate fixed
# def generate_fixed(start_words, total_num_of_words, model, id_word_dict, word_id_dict):
    
#     output = [start_words]

#     tokenised_start_words = tokenise(start_words)

#     model.eval()

#     with torch.no_grad():
    
#         for i in range(total_num_of_words):

#             current_word = torch.tensor(encode(tokenised_start_words, word_id_dict))

#             logits = model(current_word)

#             next_word_id = torch.argmax(logits)

#             next_word = id_word_dict[next_word_id.item()]

#             output.append(next_word)

#             tokenised_start_words = tokenised_start_words[1:] + [next_word]

#     return " ".join(output)

