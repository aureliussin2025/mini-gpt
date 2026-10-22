#split_txt/toeknise
def tokenise(text):
    token = []
    punctuation_list = "!,?:."

    text = text.split()

    for w in text:
        if w[-1] in punctuation_list:
            
            token.append(w[:-1])
            token.append(w[-1:])

        else:
            token.append(w)

    return token

text = "hello i'm the goat. yea!"

print(tokenise(text))