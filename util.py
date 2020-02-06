import torch  # 基本モジュール
from torch.autograd import Variable  # 自動微分用
import torch.nn as nn  # ネットワーク構築用
import torch.optim as optim  # 最適化関数
import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data  # データセット読み込み関連
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd
import numpy as np

####################### network definition #######################


class Net(nn.Module):

    def __init__(self, input_size=768, hidden_size=100, output_size=1):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.drop = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))  # F.sigmoidはdeprecated
        return x


####################### dataset definition #######################


class RESTDataset(torch.utils.data.Dataset):
    """
    標本サイズ2000
    特徴量768,カテゴリ数12
        *csvファイルには特徴量とカテゴリ、それぞれにid列が加わっているため、ilocのところで[1:]として落としている。
    """

    def __init__(self, transform=None, test=False):
        self.transform = transform
        self.data = []
        self.label = []
        df_train_x = pd.read_csv("../data/REST_train_x_vec.csv", header=None)
        df_train_y = pd.read_csv("../data/REST_train_y.csv", header=None)
        self.x = df_train_x.iloc[:, 1:].values
        self.y = df_train_y.iloc[:, 1:].values

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        out_x = self.x[idx]
        out_y = self.y[idx]

        if self.transform:
            out_x = self.transform(out_x)

        return out_x, out_y

#######################  for fine-tuning  #######################


def make_attribute_sentence(attribute: list, pre: str="", post: str="") -> list:
    tmp = []
    if pre != "" and pre[-1] != " ":
        pre = pre + " "
    if post != "" and post[0] != " ":
        post = " " + post

    for att in attribute:
        tmp.append(pre + att + post)
    return tmp


def make_bert_inputs(path="../data/REST_train_x.csv", sentence_length=128, config="bert-base-uncased", attribute=None, segmented=False, pos_change=False):
    # examples of attribute : "AMBIENCE#GENERAL", "DRINKS#PRICES",
    # "RESTAURANT#MISCELLANEOUS"
    if segmented and not attribute:
        print("\t\t\tERROR: CAN'T BE SEGMENTED BECAUSE INPUT HAVE NO ATTRIBUTE.")
        return None

    # load text from path
    text = pd.read_csv(path, header=None).iloc[:, 1:].values
    if attribute:  # make attribute input
        attribute_text = " ".join(attribute.split("#"))

    if pos_change:  # [CLS] label [SEP] text [SEP]
        if not attribute:
            print("\t\t\tERROR: ATTRIBUTE MUST BE PASSED WHEN POS_CHANGE IS TRUE.")
            return None
        text2 = "[CLS] " + attribute_text + " [SEP] " + text + " [SEP]"
    else:
        text2 = "[CLS] " + text + " [SEP]"
        if attribute:
            text2 = text2 + " " + attribute_text + " [SEP]"

    text3 = [np.array(item)[0] for item in text2]  # list of str

    # tokenize text
    tokenizer = BertTokenizer.from_pretrained(
        config, do_lower_case=True)
    tokenized_text = [tokenizer.tokenize(item) for item in text3]

    # tokenizeの結果BERT入力の最大文長を超えていないかチェック。
    for item in tokenized_text:
        if len(item) > sentence_length:
            print("\t\t\t>>>>@@@@Error sentence length is longer than expected.@@@@<<<<")
            return None

    # attribute の長さを計算
    if segmented and attribute:
        len_of_attribute = len(tokenizer.tokenize(
            attribute_text)) + 1  # [SEP]の分を足す
        if pos_change:
            len_of_attribute += 1  # [CLS]の分をたす
    else:
        len_of_attribute = 0

    input_ids = []  # list of ids [12,100,13,...,12,0,0,0] padding is 0
    segment_masks = []  # list of segment [0,0,0,0,...,0,0,0] padding is 0
    attention_masks = []  # どこがパディングか。 [1,1,...,1,0,...,0]
    #padding is 0

    for item in tokenized_text:
        len_of_item = len(item)
        len_of_text = len_of_item - len_of_attribute
        padding = [0] * (sentence_length - len_of_item)

        input_id = tokenizer.convert_tokens_to_ids(item)
        if pos_change:
            segment_mask = [0] * len_of_attribute + [1] * len_of_text
        else:
            segment_mask = [0] * len_of_text + [1] * len_of_attribute
        if not segmented:  # segmentedなら上書き
            segment_mask = [0] * len_of_item
        attention_mask = [1] * len_of_item

        input_id += padding
        segment_mask += padding
        attention_mask += padding

        input_ids.append(input_id)
        segment_masks.append(segment_mask)
        attention_masks.append(attention_mask)

    return input_ids, attention_masks, segment_masks, tokenizer


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
