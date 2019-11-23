import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, WarmupLinearSchedule
import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange  # tqdmで処理進捗を表示
from util import make_bert_inputs, flat_accuracy
from sklearn.model_selection import train_test_split

if len(sys.argv) != 5:
    print("argments must be 4.")
    print("1:cuda_num, 2:model_name, 3:sentence_length, 4:Position_reversed(0=False,1=True)")
    sys.exit(1)

cuda_num = sys.argv[1]
if sys.argv[1].isdigit() == False or sys.argv[3].isdigit() == False or sys.argv[4].isdigit() == False:
    print("argments should be numbers")
    sys.exit(1)


cuda_num = sys.argv[1]
model_name = sys.argv[2]
sentence_len = sys.argv[3]
position_reversed = bool(int(sys.argv[4]))

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

# correct label list
labels = pd.read_csv("../data/REST_test_y.csv",
                     header=None).iloc[:, 1:].values

print("label,  TP,  FP,  FN,  TN,  accuracy,  precision,  recall,  F1")

for label_num in range(0, labels.shape[1]):
    # make bert-inputs
    inputs, masks, segments, _ = make_bert_inputs(path="../data/REST_test_x.csv", sentence_length=int(
        sentence_len), config=("./tokenizer" + sentence_len), attribute=attribute_list[label_num], segmented=True, pos_change=position_reversed)
    # bert-inputs -> tensor type
    tensor_inputs = torch.tensor(inputs, requires_grad=False)
    tensor_masks = torch.tensor(masks, requires_grad=False)
    tensor_segments = torch.tensor(segments, requires_grad=False)
    input_dir = "./models_" + model_name + "/label" + str(label_num)
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
    model = BertForSequenceClassification.from_pretrained(
        input_dir, num_labels=2)
    model.to(device)
    model.eval()
    eval_accuracy = 0
    TP, FP, FN, TN = 0, 0, 0, 0
    for batch in dataloader:
        batch = [t.to(device) for t in batch]
        b_input_ids, b_input_masks, b_segments, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_masks,
                            token_type_ids=b_segments, labels=b_labels)
            logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        predict = np.argmax(logits, axis=1).flatten()
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
