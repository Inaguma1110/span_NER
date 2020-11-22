#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
get_ipython().system('test -d bertviz_repo && echo "FYI: bertviz_repo directory already exists, to pull latest version uncomment this line: !rm -r bertviz_repo"')
# !rm -r bertviz_repo # Uncomment if you need a clean pull from repo
get_ipython().system('test -d bertviz_repo || git clone https://github.com/jessevig/bertviz bertviz_repo')
if not 'bertviz_repo' in sys.path:
  sys.path += ['bertviz_repo']
get_ipython().system('pip install regex')
get_ipython().system('pip install transformers')


# In[2]:


import torch
from bertviz import head_view
from transformers import BertTokenizer, BertModel


# In[3]:


def call_html():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))


# In[19]:


model_version = 'bert-base-uncased'
do_lower_case = True
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
token_type_ids = inputs['token_type_ids']
input_ids = inputs['input_ids']
attention = model(input_ids, token_type_ids=token_type_ids)[-1]
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)
call_html()
head_view(attention, tokens)


# In[ ]:




