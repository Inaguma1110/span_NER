import tqdm, contextlib

import numpy as np
import torch
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F
import torch.utils.data    as D



import label_num_check
import preprocess
import feature


from CNN_machining import CNNs
from CNN_machining import CNNs_no_trigger


import tensorboardX as tb
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
writer = tb.SummaryWriter(log_dir ='/home/inaguma.19406/TensorboardGraph/Relation_extractionr_lr=0.001')

print('Create Environment ...')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = configparser.ConfigParser()
config.read('/home/inaguma.19406/relation/machine.conf')
trigger_flg = config.get('flg', 'TRIGGER')
parser = argparse.ArgumentParser()
arguments = parser.parse_args()
print('finish')
#pdb.set_trace()
if trigger_flg == "True":
	print('Create Data ...')
	print('Trigger flg is   ' + trigger_flg)
	#data, vocab_size, pos_size, BIO_size, trigger_size = feature.main(config)
	database = shelve.open(config.get('path', 'PATH_RETURN'))
	data, vocab_size, pos_size, BIO_size, trigger_size, rel_map =  database['feature']
	labels = database["label"]
	database.close()
	print('vocab_size, pos_size, BIO_size, trigger_size, rel_map =')
	print(vocab_size, pos_size, BIO_size, trigger_size, rel_map)
	##[sentence_word_map, pos_map, trigger_map, BIO_map, target_map, otehr_map, dbl_posi, masked_map]
	# TODO: tensor slice
	sentence_word_map   = torch.LongTensor([a[0] for a in data]).to(device)
	pos_map             = torch.LongTensor([a[1] for a in data]).to(device)
	trigger_map         = torch.LongTensor([a[2] for a in data]).to(device)
	BIO_map             = torch.LongTensor([a[3] for a in data]).to(device)
	target_map          = torch.LongTensor([a[4] for a in data]).to(device)
	other_map           = torch.LongTensor([a[5] for a in data]).to(device)
	first_position_map  = torch.LongTensor([a[6] for a in data]).to(device)
	second_position_map = torch.LongTensor([a[7] for a in data]).to(device)
	masked_map          = torch.LongTensor([a[8] for a in data]).to(device)
	#label               = torch.LongTensor([a[9] for a in data])
	tlabel              = torch.LongTensor(labels).to(device)

	max_sent_len = int(config.get('preprocess', 'INIT_MAX_SENT_LEN'))

	masked_sentence_words = sentence_word_map * masked_map
	masked_poss       = pos_map * masked_map
	masked_triggers   = trigger_map * masked_map
	masked_BIOs       = BIO_map * masked_map
	masked_target_map    = target_map * masked_map
	masked_other_map     = other_map * masked_map
	masked_1_position    = first_position_map * masked_map
	masked_2_position    = second_position_map * masked_map


	word_dim     = int(config.get('CNNs', 'WORD_DIM'))
	pos_dim      = int(config.get('CNNs', 'POS_DIM'))
	trigger_dim  = int(config.get('CNNs', 'TRIGGER_DIM'))
	BIO_dim      = int(config.get('CNNs', 'BIO_DIM'))

	dataset = D.TensorDataset(masked_sentence_words, masked_poss, masked_triggers, masked_BIOs, masked_target_map, masked_other_map, masked_1_position, masked_2_position, tlabel)

	train_size = int(0.8 * len(data))
	devel_size = int(0.1 * len(data))
	test_size  = len(data) - train_size - devel_size

	#train, devel, test = D.random_split(inputdata,[train_size, devel_size, test_size])
	train_dataset, devel_dataset, test_dataset = D.random_split(dataset, [train_size, devel_size, test_size])
	train_loader = D.DataLoader(train_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_TRAIN')), shuffle=strtobool(config.get('main', 'BATCH_SHUFFLE_TRAIN')))
	devel_loader = D.DataLoader(devel_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_DEVEL')), shuffle=strtobool(config.get('main', 'BATCH_SHUFFLE_DEVEL')))
	test_loader = D.DataLoader(test_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_TEST')), shuffle=strtobool(config.get('main', 'BATCH_SHUFFLE_TEST')))

	print('finish', end='\n\n')

	print('Create Model...')
	model = CNNs(config, vocab_size, len(rel_map), pos_size, BIO_size, trigger_size).to(device)
	print('finish', end='\n\n')

	loss_function = nn.CrossEntropyLoss()
	optimizer     = optim.Adam(model.parameters(), lr=float(config.get('main', 'LEARNING_RATE')))

	print('Start Training... ')
	start = time.time()
	model.train()




if trigger_flg == "False":
	print('Create Data ...')
	print('Trigger flg is   ' + trigger_flg)
	#data, vocab_size, pos_size, BIO_size, trigger_size = feature.main(config)
	database = shelve.open(config.get('path', 'PATH_RETURN'))
	data, vocab_size, pos_size, BIO_size, rel_map =  database['feature_no_trigger']
	labels = database["label"]
	database.close()
	print('vocab_size, pos_size, BIO_size, trigger_size, rel_map =')
	print(vocab_size, pos_size, BIO_size, rel_map)
	##[sentence_word_map, pos_map, trigger_map, BIO_map, target_map, otehr_map, dbl_posi, masked_map]
	# TODO: tensor slice
	sentence_word_map   = torch.LongTensor([a[0] for a in data]).to(device)
	pos_map             = torch.LongTensor([a[1] for a in data]).to(device)
	BIO_map             = torch.LongTensor([a[2] for a in data]).to(device)
	target_map          = torch.LongTensor([a[3] for a in data]).to(device)
	other_map           = torch.LongTensor([a[4] for a in data]).to(device)
	first_position_map  = torch.LongTensor([a[5] for a in data]).to(device)
	second_position_map = torch.LongTensor([a[6] for a in data]).to(device)
	masked_map          = torch.LongTensor([a[7] for a in data]).to(device)
	#label               = torch.LongTensor([a[9] for a in data])
	tlabel              = torch.LongTensor(labels).to(device)

	max_sent_len = int(config.get('preprocess', 'INIT_MAX_SENT_LEN'))

	masked_sentence_words = sentence_word_map * masked_map
	masked_poss       = pos_map * masked_map
	masked_BIOs       = BIO_map * masked_map
	masked_target_map    = target_map * masked_map
	masked_other_map     = other_map * masked_map
	masked_1_position    = first_position_map * masked_map
	masked_2_position    = second_position_map * masked_map


	word_dim     = int(config.get('CNNs', 'WORD_DIM'))
	pos_dim      = int(config.get('CNNs', 'POS_DIM'))
	trigger_dim  = int(config.get('CNNs', 'TRIGGER_DIM'))

	dataset = D.TensorDataset(masked_sentence_words, masked_poss, masked_BIOs, masked_target_map, masked_other_map, masked_1_position, masked_2_position, tlabel)

	train_size = int(0.8 * len(data))
	devel_size = int(0.1 * len(data))
	test_size  = len(data) - train_size - devel_size

	#train, devel, test = D.random_split(inputdata,[train_size, devel_size, test_size])
	train_dataset, devel_dataset, test_dataset = D.random_split(dataset, [train_size, devel_size, test_size])
	train_loader = D.DataLoader(train_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_TRAIN')), shuffle=strtobool(config.get('main', 'BATCH_SHUFFLE_TRAIN')))
	devel_loader = D.DataLoader(devel_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_DEVEL')), shuffle=strtobool(config.get('main', 'BATCH_SHUFFLE_DEVEL')))
	test_loader = D.DataLoader(test_dataset, batch_size=int(config.get('main', 'BATCH_SIZE_TEST')), shuffle=strtobool(config.get('main', 'BATCH_SHUFFLE_TEST')))

	print('finish', end='\n\n')

	print('Create Model...')
	model = CNNs_no_trigger(config, vocab_size, len(rel_map), pos_size, BIO_size ).to(device)
	print('finish', end='\n\n')

	loss_function = nn.CrossEntropyLoss()
	optimizer     = optim.Adam(model.parameters(), lr=float(config.get('main', 'LEARNING_RATE')))

	print('Start Training... ')
	start = time.time()
	model.train()



for epoch in range(int(config.get('main', 'N_EPOCH'))):
    print('Current Epoch:{}'.format(epoch+1))
    for i, [*xs, ys] in enumerate(tqdm.tqdm(train_loader)):
        model.zero_grad()
        logits     = model(xs)
        loss       = loss_function(logits, ys)
        loss.backward(retain_graph=True)
        optimizer.step()

    sum_loss = 0.0
    predicts = []
    answers  = []
	
    for i, [*xs, ys] in enumerate(devel_loader):

        model.eval()
        batch_size = ys.shape[0]
        logits     = model(xs)
        loss       = loss_function(logits, ys)
        sum_loss  += float(loss) * batch_size

        predicts.append(torch.max(logits, 1)[1])
        answers.append(ys)
    
    #pdb.set_trace()
    devel_re_score = U.re_scores(preds=torch.cat(predicts, 0), golds=torch.cat(answers, 0), relation_map=rel_map)
    print("    devel micro P/R/F={0:.4f}/{1:.4f}/{2:.4f} macro P/R/F={3:.4f}/{4:.4f}/{5:.4f}".format(devel_re_score["micro_precision"], devel_re_score["micro_recall"], devel_re_score["micro_f_score"], devel_re_score["macro_precision"], devel_re_score["macro_recall"], devel_re_score["macro_f_score"]),devel_re_score["individual_f_score"], "loss:", \
sum_loss)
    #writer.add_scalar('devel/f-measure/epoch', devel_re_score["micro_f_score"],epoch)
    #writer.add_scalar('devel/precission/epoch', devel_re_score["micro_precision"], epoch)
    #writer.add_scalar('devel/recall/epoch', devel_re_score["micro_recall"], epoch)
    #writer.add_scalar('devel/loss/epoch',  sum_loss, epoch)
    model.train()


    if (epoch+1)%100 == 0:

        model.eval()

        predicts = []
        answers  = []

        with torch.no_grad():
            for i, [*xs, ys] in enumerate(test_loader):

                logits     = model(xs)

                predicted  = torch.max(logits, 1)[1]

                predicts.append(torch.max(logits, 1)[1])
                answers.append(ys)

            correct = (torch.cat(predicts, 0) == torch.cat(answers, 0)).sum().item() / torch.cat(answers, 0).size(0)
            test_re_score = U.re_scores(preds=torch.cat(predicts, 0), golds=torch.cat(answers, 0), relation_map=rel_map)

            print("    test  micro P/R/F={0:.4f}/{1:.4f}/{2:.4f} macro P/R/F={3:.4f}/{4:.4f}/{5:.4f}".format(test_re_score["micro_precision"], test_re_score["micro_recall"], test_re_score["micro_f_score"], test_re_score["macro_precision"], test_re_score["macro_recall"], test_re_score["macro_f_score"]),test_re_score["individual_f_score"], "BA\
TCH:{} Test Accuracy:{}".format(i, correct))
        model.train()

    #if devel_re_score['micro_precision'] == 1.0000:
    #    pdb.set_trace()

    print('Time per Epoch:{:.1f}'.format(time.time()-start))
    start = time.time()
