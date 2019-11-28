import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, WarmupLinearSchedule
import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange  # tqdmで処理進捗を表示
from util import make_bert_inputs, flat_accuracy, make_attribute_sentence
from sklearn.model_selection import train_test_split
import torch.nn as nn

torch.manual_seed(2019)

if len(sys.argv) != 7:
    print("argments must be 6.")
    print("1:cuda_num, 2:start_label, 3:sentence length, 4:Position_reversed(0=False,1=True), 5: if use weight 1/not correctly implemented, 6:model_name")
    sys.exit(1)

if sys.argv[1].isdigit() == False or sys.argv[2].isdigit() == False or sys.argv[3].isdigit == False or sys.argv[4].isdigit == False or sys.argv[5].isdigit == False:
    print("argments should be numbers")
    sys.exit(1)

cuda_num = sys.argv[1]
start_label = int(sys.argv[2])
sentence_len = int(sys.argv[3])
position_reversed = bool(int(sys.argv[4]))
use_weight = bool(int(sys.argv[5]))
model_name = sys.argv[6]

if torch.cuda.is_available():
    device = torch.device('cuda:' + cuda_num)
else:
    device = torch.device('cpu')

#######################  setting  #######################
# for train loop
epoch_size = 10
batch_size = 30
# for warmup schedule
num_total_steps = epoch_size * batch_size
num_warmup_steps = num_total_steps * 0.1
# for gradient clipping
max_grad_norm = 1.0
#########################################################
attribute_list = ["AMBIENCE#GENERAL", "DRINKS#PRICES", "DRINKS#QUALITY", "DRINKS#STYLE_OPTIONS", "DRINKS#STYLE_OPTIONS", "FOOD#PRICES",
                  "FOOD#STYLE_OPTIONS", "LOCATION#GENERAL", "RESTAURANT#GENERAL", "RESTAURANT#MISCELLANEOUS", "RESTAURANT#PRICES", "SERVICE#GENERAL"]

labels = pd.read_csv("../data/REST_train_y.csv",
                     header=None).iloc[:, 1:].values

all_ids, all_masks, all_segment_masks = [], [], []

for label_num in trange(start_label, labels.shape[1], desc="make-Label"):
    # make bert-inputs and correct label list
    ids, masks, segment_masks, tokenizer = make_bert_inputs(
        path="../data/REST_train_x.csv", sentence_length=sentence_len, attribute=attribute_list[label_num], segmented=True, pos_change=position_reversed)
    all_ids.extend(ids)
    all_masks.extend(masks)
    all_segment_masks.extend(segment_masks)
os.makedirs("./tokenizer_att3_" + str(sentence_len), exist_ok=True)
tokenizer.save_pretrained("./tokenizer_att3_" + str(sentence_len))
del tokenizer


# split inputs and labels into 1.)train data & 2.)validation data
thelabel = labels.T.reshape(1, -1)[0]
train_inputs, train_labels, train_masks, train_segment = all_ids, thelabel, all_masks, all_segment_masks

# weightを作る用
if use_weight:
    positive_size = np.count_nonzero(thelabel)
    negative_size = len(thelabel) - positive_size
    pos_weight = len(thelabel) / (2 * (positive_size + 10))
    neg_weight = len(thelabel) / (2 * (negative_size + 10))

# bert-inputs & label -> tensor type
train_inputs = torch.tensor(train_inputs, requires_grad=False)
train_labels = torch.tensor(train_labels, requires_grad=False)
train_masks = torch.tensor(train_masks, requires_grad=False)
train_segment = torch.tensor(train_segment, requires_grad=False)

# make dataloader
train_data = TensorDataset(train_inputs, train_masks,
                           train_labels, train_segment)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=batch_size)

# prepare bert model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2)
model.to(device)

# prepare optimizer and scheduler
param_optimizer = list(model.named_parameters())
no_decay = ["bias", "gamma", "beta"]
optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01}, {
    'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters,
                  lr=2e-5, correct_bias=False)
scheduler = WarmupLinearSchedule(
    optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)

for _ in trange(epoch_size, desc="Epoch"):
    tr_loss = 0
    nb_tr_steps = 0

    model.train()
    for batch in train_dataloader:
        batch = [t.to(device) for t in batch]
        b_input_ids, b_input_masks, b_labels, b_segments = batch
        optimizer.zero_grad()
        outputs = model(b_input_ids,
                        attention_mask=b_input_masks, labels=b_labels, token_type_ids=b_segments)
        if use_weight:
            temp = b_labels.cpu().numpy()
            weight = np.where(temp == 1, pos_weight, neg_weight)
            m = nn.Softmax(dim=1)
            criterion = nn.modules.BCELoss(
                weight=torch.from_numpy(weight).float().to(device))
            logits = outputs[1]
            predicts = m(logits)[:, -1]
            loss = criterion(predicts.to(device), b_labels.float())
        else:
            loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            optimizer_grouped_parameters[0]["params"], max_grad_norm)
        torch.nn.utils.clip_grad_norm_(
            optimizer_grouped_parameters[1]["params"], max_grad_norm)
        scheduler.step()
        optimizer.step()
        tr_loss += float(loss.item())
        nb_tr_steps += 1
    tqdm.write("Train loss: {}".format(tr_loss / nb_tr_steps))
    model.eval()
    del outputs
output_dir = "./models_" + model_name + "/label" + "_att3"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model.save_pretrained(output_dir)
