############ index to texts ##########
def index2text(words):
    sentences = [a for a in words]
    predicted_tokens = []
    for s_num in range(0, words.size(0)):
        one_predicted_tokens = []
        for i_num in range(0, words.size(1)):
            predicted_token = tokenizer.convert_ids_to_tokens(sentences[s_num][i_num].item())
            one_predicted_tokens.append(predicted_token)
        predicted_tokens.append(one_predicted_tokens)