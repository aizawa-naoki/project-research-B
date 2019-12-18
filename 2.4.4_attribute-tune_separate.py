import argparse
from distutils.util import strtobool
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


#------------------------parser------------------------
parser = argparse.ArgumentParser(
    description="attributeごとに別々のmodelを作成して、modelをfine-tuneするプログラム")

parser.add_argument("cuda", help="使用するGPUの番号を指定してください。0以上の整数値です。")
parser.add_argument("model_name", help="保存する際に使用するモデルの名前を指定してください。")
parser.add_argument(
    "polarity", help="極性を含めた分類をする(True)か否か(False)です。", type=strtobool)
parser.add_argument(
    "--reversed", help="attributeを入力する際に[文、属性](False)の順で渡すか、[属性、文](True)の順で渡すかを指定できます", type=strtobool, default=0)
parser.add_argument(
    "--segmented", help="attributeを入力する2文に明示的に分けるか、同じ文章として入力するか指定します", type=strtobool, default=0)
parser.add_argument(
    "--weighted", help="正例と不例の量の不均衡を補うために、逆伝搬の重みづけを行います", type=strtobool, default=1)
parser.add_argument("--sentence_length",
                    help="tokenizerに渡す文長を指定してください。", type=int, default=120)

parser.add_argument(
    "--start_label", help="fine-tuneを始めるattributeの番号を指定してください", type=int, default=0)
parser.add_argument(
    "--pre", help="Q&A形式にするためにattributeの\"前\"に追加する文を入力してください", default="")
parser.add_argument(
    "--post", help="Q&A形式にするためにattributeの\"後\"に追加する文を入力してください", default="")
parser.add_argument("--epoch", help="訓練のエポック数を指定してください", type=int, default=6)

args = parser.parse_args()

#----------------------parser_end----------------------

#----------------------import args---------------------
cuda_num = args.cuda
start_label = args.start_label
polarity = bool(args.polarity)
sentence_len = args.sentence_length
position_reversed = bool(args.reversed)
segmented = bool(args.segmented)
use_weight = bool(args.weighted)
model_name = args.model_name
pre = args.pre
post = args.post
#----------------------import end----------------------

if torch.cuda.is_available():
    device = torch.device('cuda:' + cuda_num)
else:
    device = torch.device('cpu')

#######################  setting  #######################
# for train loop
epoch_size = args.epoch
batch_size = 10
# for warmup schedule
num_total_steps = epoch_size * batch_size
num_warmup_steps = num_total_steps * 0.1
# for gradient clipping
max_grad_norm = 1.0
#########################################################

attribute_list = ["AMBIENCE#GENERAL", "DRINKS#PRICES", "DRINKS#QUALITY", "DRINKS#STYLE_OPTIONS", "DRINKS#STYLE_OPTIONS", "FOOD#PRICES",
                  "FOOD#STYLE_OPTIONS", "LOCATION#GENERAL", "RESTAURANT#GENERAL", "RESTAURANT#MISCELLANEOUS", "RESTAURANT#PRICES", "SERVICE#GENERAL"]

attribute_list = make_attribute_sentence(attribute_list, pre=pre, post=post)

if polarity:
    labels = pd.read_csv("..data/REST_train_y_polarity.csv",
                         header=None).iloc[:, 1:].values
else:
    labels = pd.read_csv("../data/REST_train_y.csv",
                         header=None).iloc[:, 1:].values


for label_num in trange(start_label, labels.shape[1], desc="Label"):
    # make bert-inputs and correct label list
    ids, masks, segment_masks, tokenizer = make_bert_inputs(
        path="../data/REST_train_x.csv", sentence_length=sentence_len, attribute=attribute_list[label_num], segmented=segmented, pos_change=position_reversed)
    os.makedirs("./tokenizer" + str(sentence_len), exist_ok=True)
    tokenizer.save_pretrained("./tokenizer" + str(sentence_len))
    del tokenizer

    # split inputs and labels into 1.)train data & 2.)validation data
    thelabel = labels[:, label_num]
    train_inputs, train_labels, train_masks, train_segment = ids, thelabel, masks, segment_masks

    # weightを作る用
    if use_weight:
        reactive_size = np.count_nonzero(train_labels)
        neutral_size = len(train_labels) - reactive_size
        react_weight = len(train_labels) / (2 * (reactive_size + 10))
        neutr_weight = len(train_labels) / (2 * (neutral_size + 10))
        # TODO: use_weight節の中身を3値分類しても機能するように拡張

    # 極性ある場合はlabelの値を0,1,2とする
    # negative,neutral,positive = -1,0,1 から 2,0,1に置換
    #（model.forwardの際に、torch.nn.CrossEntropyLossにlabelが送られるが、そこで0以上の連続する自然数と指定されているため。）
    if polarity:
        train_labels = np.where(train_labels == -1, 2, train_labels)

    # bert-inputs & label -> tensor type
    train_inputs = torch.tensor(train_inputs, requires_grad=False)
    train_labels = torch.tensor(train_labels, requires_grad=False)
    train_masks = torch.tensor(train_masks, requires_grad=False)
    train_segment = torch.tensor(train_segment, requires_grad=False)

    # make dataloader
    train_data = TensorDataset(
        train_inputs, train_masks, train_labels, train_segment)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size)

    # set num_labels
    if polarity:
        num_labels = 3
    else:
        num_labels = 2

    # prepare bert model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels)
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
                if polarity:
                    weight = torch.tensor(
                        [neutr_weight, react_weight, react_weight], requires_grad=False, device=device)
                    criterion = nn.modules.CrossEntropyLoss(
                        weight=torch.from_numpy(weight).float().to(device))
                    logits = outputs[1]
                    loss = criterion(logits.view(-1, num_labels),
                                     b_labels.view(-1))
                else:
                    temp = b_labels.cpu().numpy()
                    weight = np.where(temp == 0, neutr_weight, react_weight)
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
    output_dir = "./models_" + model_name + "/label" + str(label_num)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    model.to("cpu")
    batch = [t.to("cpu") for t in batch]
    del train_inputs
    del train_labels
    del train_masks
    del train_data
    del train_sampler
    del train_dataloader
    del model
    del batch
    del b_input_ids
    del b_input_masks
    del b_labels
    del b_segments
    del loss
    del nb_tr_steps
    del no_decay
    del optimizer_grouped_parameters
    del param_optimizer
    del optimizer
    del scheduler
    del thelabel
    del tr_loss
