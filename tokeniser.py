#split_txt/tokenise
def tokenise(text):
    token = []
    punctuation_list = "!,?:.;"

    text = text.split()

    for w in text:
        if w[-1] in punctuation_list:
            
            token.append(w[:-1])
            token.append(w[-1:])

        else:
            token.append(w)

    return token


def create_vocab(token):
    id = 0
    word_id_dict = {}

    for w in token:

        if word_id_dict.get(w) == None:
            word_id_dict[w] = id
            id += 1

    id_word_dict = {value: key for key, value in word_id_dict.items()}

    return word_id_dict, id_word_dict

def encode(token, word_id_dict):
    encoded_list = []

    for t in token:
        encoded_list.append(word_id_dict[t])

    return encoded_list

def decode(encoded_list, id_word_dict):
    decoded_list = ""
    
    for id in encoded_list:
        decoded_list += (id_word_dict.get(id) + " ")
    return decoded_list