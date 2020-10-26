def predtotext(words,b_num,n_doc,n_docs,tokenizer,logits_span1,logits_span2,logits_span3,logits_span4):
    predicted_tokens = []
    for_txt_span1 = []
    for_txt_span2 = []
    for_txt_span3 = []
    for_txt_span4 = []
    sentences = [a for a in words]
    one_predicted_tokens = []
    one_for_txt_span1 = [] 
    one_for_txt_span2 = []
    one_for_txt_span3 = []
    one_for_txt_span4 = []
    for i_num in range(0, words.size(1)):
        predicted_token = tokenizer.convert_ids_to_tokens(sentences[b_num][i_num].item())
        one_predicted_tokens.append(predicted_token)
        one_for_txt_span1.append(torch.max(logits_span1, 1)[1][b_num][i_num].item())
        one_for_txt_span2.append(torch.max(logits_span2, 1)[1][b_num][i_num].item())
        one_for_txt_span3.append(torch.max(logits_span3, 1)[1][b_num][i_num].item())
        one_for_txt_span4.append(torch.max(logits_span4, 1)[1][b_num][i_num].item())
    predicted_tokens.append(one_predicted_tokens)
    for_txt_span1.append(one_for_txt_span1)
    for_txt_span2.append(one_for_txt_span2)
    for_txt_span3.append(one_for_txt_span3)
    for_txt_span4.append(one_for_txt_span4)
    n_docs.append(n_doc[b_num][0].item())

    return predicted_tokens,n_docs
