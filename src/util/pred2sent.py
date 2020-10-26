import pdb
import torch


def pred2sent(sents, WORD_DIC, preds, TAG_DIC):
    rslts = []
    inverse_WORD_DIC = {v:k for k,v in WORD_DIC.items()}
    inverse_TAG_DIC  = {v:k for k,v in TAG_DIC.items()}
    #import pdb;pdb.set_trace()
    for words in sents:
        text = []
        tag = []
        for j, word_id in enumerate(words):
            if word_id.item() == 0:
                continue
            text.append(inverse_WORD_DIC[word_id.item()])

            if preds[0][0][j].item() == 0:
                continue
            tag.append(inverse_TAG_DIC[preds[0][0][j].item()])

    for rslt in zip(text, tag):
        rslts.append(rslt)
    return rslts
