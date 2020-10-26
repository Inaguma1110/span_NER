import sys
import argparse
import pdb
import tqdm, contextlib
import time
import shelve
import configparser
from progressbar import progressbar
from distutils.util import strtobool
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as D
from torch.utils.tensorboard import SummaryWriter

import tensorboardX as tb

from model.BRAN import BERT_PRETRAINED_MODEL_JAPANESE, SPAN_CNN, PAIRS_MODULE, RELATION, BRAN
from util import pred2ann,pred2text,nest_entity_process

config = configparser.ConfigParser()
config.read('../machine_BRAN.conf')