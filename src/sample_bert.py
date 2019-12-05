import torch
import sentencepiece as spm
import pdb
import collections
import tensorflow as tf
import six

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig, BertForSequenceClassification


sp = spm.SentencePieceProcessor()
sp.Load('./spm_model/wiki-ja.model')
tokenizer = BertTokenizer.from_pretrained('./spm_model/wiki-ja.vocab.txt')

spmed = sp.EncodeAsPieces('切削抵抗は、切りくずの形で材料を除去するのに必要な切削力の反力で、長手削りの場合の図1.1-1でみるように、切削合力Fは主分力F_v、送り分力F_fおよび背分力F_pの3分力に分解できる。')
#spmed_ = tokenizer.tokenize(' '.join(spmed))
#indexed_tokens = tokenizer.convert_tokens_to_ids(spmed_)
#for i, idx in enumerate(indexed_tokens):
#    if idx == None:
#        pdb.set_trace()

#tokens_tensor = torch.tensor([indexed_tokens])

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")
    
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token, _ = token.split("\t")
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

vocab = load_vocab('./spm_model/wiki-ja.vocab')
mask_indx = 12

pdb.set_trace()
spmed[0] = '[CLS]'
spmed.append('[SEP]')
spmed[mask_indx] = '[MASK]'

indx_tokens = [vocab[s] if s in vocab else vocab['<unk>'] for s in spmed]

tokens_tensor = torch.tensor([indx_tokens])

config = BertConfig.from_json_file('../published_model/bert_spm/bert_config.json')
model = BertModel.from_pretrained('../published_model/bert_spm/pytorch_model.bin',config=config)
model2 = BertForMaskedLM.from_pretrained('../published_model/bert_spm/pytorch_model.bin',config=config)
model3 = BertModel.from_pretrained('../published_model/bert_spm/pytorch_model.bin',config=config)

model.eval()
model2.eval()
model3.eval()

tokens_tensor = tokens_tensor.to('cuda')

model.to('cuda')
model2.to('cuda')
model3.to('cuda')

pdb.set_trace()
with torch.no_grad():
    all_encoder_layers = model(tokens_tensor)
    outputs = model2(tokens_tensor)
    outputs3= model3(tokens_tensor)
    predictions = outputs[0]

pdb.set_trace()

pooling_layer = -2
embedding = all_encoder_layers[pooling_layer]


_, predicted_indexes = torch.topk(predictions[0, mask_indx], k=15)
predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())


