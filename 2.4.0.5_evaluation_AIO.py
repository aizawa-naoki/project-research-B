import argparse
from distutils.util import strtobool
import torch
from transformers import BertTokenizer, BertModel, AdamW, WarmupLinearSchedule
import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange  # tqdmで処理進捗を表示
from util import make_bert_inputs, flat_accuracy, make_attribute_sentence, Net
from sklearn.model_selection import train_test_split

#------------------------parser------------------------
parser = argparse.ArgumentParser(
    description="attributeごとに別々のmodelを読み込んで、evaluationするプログラム")

parser.add_argument("cuda", help="使用するGPUの番号を指定してください。0以上の整数値です。")
parser.add_argument("model_name", help="保存する際に使用するモデルの名前を指定してください。")
parser.add_argument("--sentence_length",
                    help="tokenizerに渡す文長を指定してください。", type=int, default=120)
parser.add_argument(
    "--reversed", help="attributeを入力する際に[文、属性](False)の順で渡すか、[属性、文](True)の順で渡すかを指定できます", type=strtobool, default=0)
parser.add_argument(
    "--segmented", help="attributeを入力する2文に明示的に分けるか、同じ文章として入力するか指定します", type=strtobool, default=0)
parser.add_argument(
    "--pre", help="Q&A形式にするためにattributeの\"前\"に追加する文を入力してください", default="")
parser.add_argument(
    "--post", help="Q&A形式にするためにattributeの\"後\"に追加する文を入力してください", default="")
parser.add_argument("--epoch", help="訓練のエポック数を指定してください", type=int, default=6)


args = parser.parse_args()

#----------------------parser_end----------------------

#----------------------import args---------------------
cuda_num = args.cuda
model_name = args.model_name
sentence_len = args.sentence_length
position_reversed = bool(args.reversed)
segmented = bool(args.segmented)
pre = args.pre
post = args.post
#----------------------import end----------------------

if torch.cuda.is_available():
    device = torch.device('cuda:' + cuda_num)
else:
    device = torch.device('cpu')

#######################  setting  #######################
batch_size = 30
#########################################################

attribute_list = ["AMBIENCE#GENERAL", "DRINKS#PRICES", "DRINKS#QUALITY", "DRINKS#STYLE_OPTIONS", "DRINKS#STYLE_OPTIONS", "FOOD#PRICES",
                  "FOOD#STYLE_OPTIONS", "LOCATION#GENERAL", "RESTAURANT#GENERAL", "RESTAURANT#MISCELLANEOUS", "RESTAURANT#PRICES", "SERVICE#GENERAL"]

attribute_list = make_attribute_sentence(attribute_list, pre=pre, post=post)

# correct label list
labels = pd.read_csv("../data/REST_test_y.csv",
                     header=None).iloc[:, 1:].values

print("label,  TP,  FP,  FN,  TN,  accuracy,  precision,  recall,  F1")

for label_num in range(0, labels.shape[1]):
    # make bert-inputs
    inputs, masks, segments, _ = make_bert_inputs(path="../data/REST_test_x.csv", sentence_length=sentence_len, config=(
        "./tokenizer_att3_" + str(sentence_len)), attribute=attribute_list[label_num], segmented=segmented, pos_change=position_reversed)
    # bert-inputs -> tensor type
    tensor_inputs = torch.tensor(inputs, requires_grad=False)
    tensor_masks = torch.tensor(masks, requires_grad=False)
    tensor_segments = torch.tensor(segments, requires_grad=False)
    input_dir = "./models_" + model_name + "/label" + "_att3"
    if not os.path.exists(input_dir):
        print("Error: input_dir not exist.")

    # split inputs and labels into 1.)train data & 2.)validation data
    thelabel = labels[:, label_num]
    tensor_labels = torch.tensor(thelabel, requires_grad=False)

    # make dataloader
    dataset = TensorDataset(tensor_inputs, tensor_masks,
                            tensor_segments, tensor_labels)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size)

    # prepare bert model
    model = BertModel.from_pretrained(input_dir)
    model.to(device)
    model.eval()
    head = Net()
    head.load_state_dict(torch.load(input_dir + "/head"))
    head.to(device)
    head.eval()
    eval_accuracy = 0
    TP, FP, FN, TN = 0, 0, 0, 0
    for batch in dataloader:
        batch = [t.to(device) for t in batch]
        b_input_ids, b_input_masks, b_segments, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_masks,
                            token_type_ids=b_segments)
            bert_out = torch.mean(outputs[0], 1)
            head_out = torch.mean(head(bert_out), 1)
        logits = head_out.detach().cpu().numpy()
        predict = np.where(logits > 0.5, 1, 0)
        label_ids = b_labels.to("cpu").numpy()
        answer = label_ids.flatten()
        TP += np.count_nonzero((predict == 1) & (answer == 1))
        FP += np.count_nonzero((predict == 1) & (answer == 0))
        FN += np.count_nonzero((predict == 0) & (answer == 1))
        TN += np.count_nonzero((predict == 0) & (answer == 0))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)
    print("%5d,%4d,%4d,%4d,%4d,%10f,%11f,%8f,%4f" %
          (label_num + 1, TP,  FP,  FN,  TN, accuracy, precision,  recall,  F1))
    del model
    del head
