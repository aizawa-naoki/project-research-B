import torch
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
import csv
import os
import numpy as np
import ast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import logging
import configparser
from tqdm import tqdm, trange  # tqdmで処理進捗を表示
from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, confusion_matrix
import pickle
from emo_PRF import eval_multi_label, fail_and_success, multi_classification_report
#logger = logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTforMultiLabelClassification(BertPreTrainedModel):
    '''
    input_featureをBERTに入力
    ->出力からdrop_out_rateで選択された値だけLinear層に入力
    ->線形変換によりshape=(num_label,)に変換
    '''

    def __init__(self, config, num_labels=19):
        super(BERTforMultiLabelClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)  # BERTのパラメータを修正？

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:  # training case
            loss_function = torch.nn.BCEWithLogitsLoss()
            loss = loss_function(
                logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:  # test case
            return logits

    def freeze_bert_encoder(self):  # bertのパラメータを固定する関数？finetuneなしはこの関数呼びました
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeature(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def loaddata_and_create_examples(set_type, config):
    '''set_type: train or dev
    '''
    data_dir = config['path']['data_dir'] + \
        config['path']['{}_data'.format(set_type)]

    with open(data_dir, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lines = []
        for line in reader:
            for i in range(5):
                line[i] = line[i].strip()
            lines.append(line)

    examples = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        if config['data'].getboolean('with_context'):
            if line[1].find('No Context') == -1:  # contextがあるなら
                line[1] = line[1].replace('.', '. [SEP]')
                text_a = line[1] + line[2]
                #text_a = line[1] + ' ' + line[2]
            else:
                text_a = line[2]
        else:
            text_a = line[2]

        if config['data'].getboolean('with_char#'):
            text_a = line[3] + '#' + text_a
        label = ast.literal_eval(line[-1])
        examples.append(InputExample(
            guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def convert_examples_to_features(examples, max_seq_len, tokenizer):
    '''
    example -> list of InputBatchesに変換
    input_ids: max_seq_lenに文長をあわせ(切り捨てor padding)，tokenize(word->idxに変換)したもの.
    input_mask: input_idsの各indexがpadding->0, otherwise->1
    segment_ids: ブログ記事のように文章が2文以上の際文の切れ目を指定。input_idsのそれぞれの単語に、1文目ならば0を、2文目ならば1を...というように対応づけたフラグのベクトル
    label_id:　exampleのlabel_idそのまま ex.[0,0,0,...1,0] (len=19)

    Ex)
    max_seq_len =10
    tokenizer = なんでも
    examples.text_a = ["i","am","a","cat","."]
    ->
    input_ids =      [   1, 100, 11,   12,  4, 0, 0, 0, 0, 0] (単語を単語番号に置換。paddingをパディング番号(=0)に置換)
    input_mask =     [   1,   1,  1,    1,  1, 0, 0, 0, 0, 0]
    segment_ids =    [   0,   0,  0,    0,  0, 0, 0, 0, 0, 0]
    label_id    =    [label for label in labels] <文に対応するマルチラベル>
    '''

    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Now writing ex.%d of %d" % (ex_index, len(examples)))
        tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > max_seq_len - 2:
            tokens_a = tokens_a[:(max_seq_len - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # mask 1:real tokens, 0:padding tokens
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        label_id = example.label

        features.append(
            InputFeature(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id
                         ))
    return features

if __name__ == '__main__':
    # load config parser
    config = configparser.ConfigParser()
    # config読み込んでた

    # set defalt valiables
    #do_trainはTrue, max_seq_len, tokenizer, train_batch_size, gradient_accumulation_steps, num_train_epochsを決めた
    do_train, max_seq_len, tokenizer, train_batch_size, gradient_accumulation_steps, num_train_epochs =
    config['learning'].getboolean('do_train'), int(config['learning']['max_seq_length']), BertTokenizer.from_pretrained(config['learning']['bert_model']), int(
        config['learning']['train_batch_size']), int(config['learning']['gradient_accumulation_steps']), int(config['learning']['num_train_epochs'])

    # cudaのハナシ TODO gpu3個とかの指定可能な形に修正
#    local_rank = [0, 1, 2]
    # -1: no_cuda=True or 使えるGPU=nullじゃなければGPUフルで使う
    local_rank = int(config['learning']['local_rank'])
    no_cuda = config['learning'].getboolean('no_cuda')
    num_labels = int(config['learning']['num_labels'])

    if local_rank == -1 or no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # n_gpu = len(local_rank) #TODO gpu複数個の設定
        n_gpu = 1

    if do_train:
        if local_rank in [-1, 0]:
            tb_writer = SummaryWriter()
        # Load data
        train_examples = loaddata_and_create_examples("train", config)
        # convert to features
        train_features = convert_examples_to_features(
            train_examples, max_seq_len, tokenizer)

        # change dtype to torch.tensor
        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # if distributed sampling is needed, change to
        # DistributedSampler(train_data)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=train_batch_size)
        num_train_optimization_steps = \
            len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

# ここまで読んだ

        # call the model
        #bertconfig = BertConfig(vocab_size=32000)
        model = BERTforMultiLabelClassification.from_pretrained(
            config['learning']['bert_model'], num_labels=num_labels)
        if config['model'].getboolean('freeze_bert_encoder'):
            model.freeze_bert_encoder()
        model.to(device)
        if local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[
                                                                  local_rank],
                                                              output_device=local_rank,
                                                              find_unused_parameters=True)

       # prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=float(config['learning']['learning_rate']),
                             warmup=float(config['learning'][
                                          'warmup_propotion']),
                             t_total=num_train_optimization_steps)
        # TODO write logger.info here.

        model.train()
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0

        #### Training ###
        for _ in trange(int(config['learning']['num_train_epochs']), desc="Epoch", disable=local_rank not in [-1, 0]):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                             labels=label_ids)
                if n_gpu > 1:
                    loss = loss.mean()
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward(loss)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if local_rank in [-1, 0]:
                        tb_writer.add_scalar(
                            'lr', optimizer.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', loss.item(), global_step)
        tb_writer.close()
    # saving best-practices
    if do_train and (local_rank == -1 or torch.distributed.get_rank() == 0):
        model_to_save = model.module if hasattr(model, 'module') else model
        # saveしたモデルはfrom_pretrainedで使える
        output_dir = config['path']['output_dir']
        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(output_dir)
        # for config.ini
        with open(output_dir + 'config.ini', 'w') as configfile:
            config.write(configfile)

        # Reload the trained model and vocab that u have fine-tuned
        model = BERTforMultiLabelClassification.from_pretrained(
            output_dir, num_labels=int(config['learning']['num_labels']))
        tokenizer = BertTokenizer.from_pretrained(
            output_dir,
            do_lower_case=config['learning'].getboolean('do_lower_case'))
        print("Done saving models")
    else:
        # model = BERTforMultiLabelClassification.from_pretrained(
        #                                        config['learning']['bert_model'],
        #                                        num_labels=int(config['learning']['num_labels']))
        model = BERTforMultiLabelClassification.from_pretrained(
            config['path']['output_dir'], num_labels=int(config['learning']['num_labels']))

    model.to(device)

    # Eval
    if config['learning'].getboolean('do_eval') and (local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = loaddata_and_create_examples(
            set_type="dev", config=config)
        eval_features = convert_examples_to_features(
            eval_examples, max_seq_len, tokenizer)

        #import pdb; pdb.set_trace()
        logger.info("**** Running Eveluation ***")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", int(
            config['learning']['eval_batch_size']))

        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # if distributed sampling is needed, change to
        # DistributedSampler(train_data)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data,
                                     sampler=eval_sampler,
                                     batch_size=int(config['learning']['eval_batch_size']))
        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        out_label_ids = None
        sentences = []

        for nb_eval_steps, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                logits = model(input_ids, token_type_ids=segment_ids,
                               attention_mask=input_mask)
                logits = logits.sigmoid()  # ここいるかわらからんけど

            loss_function = torch.nn.BCEWithLogitsLoss()
            tmp_eval_loss = loss_function(
                logits.view(-1, num_labels), label_ids.view(-1, num_labels))
            eval_loss += tmp_eval_loss.mean().item()

            input_ids = input_ids.detach().cpu().tolist()
            for token_id in input_ids:
                sentences.append(tokenizer.convert_ids_to_tokens(token_id))

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
                out_label_ids = label_ids.detach().cpu().numpy()
            else:
                # よくわからんけどそのままappendするとミスるらしい
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, label_ids.detach().cpu().numpy(), axis=0
                                          )
        eval_loss = eval_loss / nb_eval_steps
        preds = np.round(preds[0])  # TODO 丸め誤差問題

        P, R, F1 = eval_multi_label(out_label_ids, preds)
        print("P:{}".format(P))
        print("R:{}".format(R))
        print("F1:{}".format(F1))
        if config['learning'].getboolean('do_char_eval'):
            with open(config['path']['output_dir'] + 'eval_result_char.csv', 'w') as f:
                columns = ['P', 'R', 'F1']
                results = [np.round(P, 4), np.round(R, 4), np.round(F1, 4)]

                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerow(results)
                writer.writerow([' '])
        #       print("Done writing evaluation result.")

            with open(config['path']['output_dir'] + 'preds_char.pickle', 'wb') as f:
                pickle.dump(preds, f)
            with open(config['path']['output_dir'] + 'out_label_ids_char.pickle', 'wb') as f:
                pickle.dump(out_label_ids, f)
            with open(config['path']['output_dir'] + 'sentences_char.pickle', 'wb') as f:
                pickle.dump(sentences, f)
        else:
            with open(config['path']['output_dir'] + 'eval_result.csv', 'w') as f:
                columns = ['P', 'R', 'F1']
                results = [np.round(P, 4), np.round(R, 4), np.round(F1, 4)]

                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerow(results)
                writer.writerow([' '])

            with open(config['path']['output_dir'] + 'preds.pickle', 'wb') as f:
                pickle.dump(preds, f)
            with open(config['path']['output_dir'] + 'out_label_ids.pickle', 'wb') as f:
                pickle.dump(out_label_ids, f)
            with open(config['path']['output_dir'] + 'sentences.pickle', 'wb') as f:
                pickle.dump(sentences, f)
