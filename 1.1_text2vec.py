from bert_serving.client import BertClient
import numpy as np
import pandas as pd
"""
bert-serving-start -model_dir ~/bert/uncased_L-12_H-768_A-12/ -num_worker=4 -port_in 5557 -port_out 5558 -max_seq_len 80
"""

data = pd.read_csv("../data/REST_test_x.csv", header=None)
print("\n------------DATA BELOW---------------")
print(data.head())
text = data[1]
print("\n------------TEXT BELOW---------------")
print(text.head())
que_text = text.values.tolist()

# bc = BertClient(ip="localhost", port=5557, port_out=5558)

with BertClient(ip="localhost", port=5557, port_out=5558) as bc:
    vectors = bc.encode(que_text)

print(vectors[0, :5])
