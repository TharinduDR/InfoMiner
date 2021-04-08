import shutil

import pandas as pd
import numpy as np
import os

import torch
from sklearn.model_selection import train_test_split

from examples.common.converter import encode, decode
from examples.common.evaluation import precision, recall, f1, confusion_matrix_values
from examples.bulgarian.bb_mbert.bul_bb_mbert_config import TEMP_DIRECTORY, config, MODEL_TYPE, MODEL_NAME, SEED, \
    SUBMISSION_FILE
from examples.sample_size_counter import sample_size_counter
from infominer.classification import ClassificationModel

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

train = pd.read_csv(os.path.join("examples", "bulgarian", "data", "covid19_disinfo_binary_bulgarian_train.tsv"), sep='\t')

train.dropna(subset=["q6_label"], inplace=True)

# Class count
count_class_no, count_class_yes = train.q6_label.value_counts().sort_index(ascending=True)

# Divide by class
df_class_no = train[train['q6_label'] == "no"]
df_class_yes = train[train['q6_label'] == "yes"]

size_counter = sample_size_counter(count_class_no, count_class_yes, 2)
print("NOs : ", df_class_no['q6_label'].count())
print("YESs : ", df_class_yes['q6_label'].count())
print("size counter : ", size_counter)

if size_counter > 0:

    df_class_no_under = df_class_no.sample(count_class_yes * size_counter)
    print("under sized NOs : ", df_class_no_under['q6_label'].count())
    train = pd.concat([df_class_no_under, df_class_yes], axis=0)

else:

    df_class_yes_under = df_class_yes.sample(count_class_no * abs(size_counter))
    print("under sized YESs : ", df_class_yes_under['q6_label'].count())
    train = pd.concat([df_class_yes_under, df_class_no], axis=0)

train['labels'] = encode(train["q6_label"])
train = train[['text', 'labels']]

dev = pd.read_csv(os.path.join("examples", "bulgarian", "data", "covid19_disinfo_binary_bulgarian_dev.tsv"), sep='\t')

dev.dropna(subset=["q6_label"], inplace=True)
dev['labels'] = encode(dev["q6_label"])
dev = dev[['text', 'labels']]

test = pd.read_csv(os.path.join("examples", "bulgarian", "data", "covid19_disinfo_binary_bulgarian_test_input.tsv"), sep='\t')

dev_sentences = dev['text'].tolist()
dev_preds = np.zeros((len(dev_sentences), config["n_fold"]))

test_sentences = test['text'].tolist()
test_preds = np.zeros((len(test_sentences), config["n_fold"]))

for i in range(config["n_fold"]):
    if os.path.exists(config['output_dir']) and os.path.isdir(config['output_dir']):
        shutil.rmtree(config['output_dir'])
    print("Started Fold {}".format(i))
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=config,
                                use_cuda=torch.cuda.is_available())
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
    model.train_model(train_df, eval_df=eval_df, precision=precision, recall=recall, f1=f1)
    model = ClassificationModel(MODEL_TYPE, config["best_model_dir"], args=config,
                                use_cuda=torch.cuda.is_available())

    predictions, raw_outputs = model.predict(dev_sentences)
    dev_preds[:, i] = predictions

    test_predictions, test_raw_outputs = model.predict(test_sentences)
    test_preds[:, i] = test_predictions

    print("Completed Fold {}".format(i))

# select majority class of each instance (row)
dev_predictions = []
for row in dev_preds:
    row = row.tolist()
    dev_predictions.append(int(max(set(row), key=row.count)))


# select majority class of each instance (row)
test_predictions = []
for row in test_preds:
    row = row.tolist()
    test_predictions.append(int(max(set(row), key=row.count)))


dev["predictions"] = dev_predictions


print("Precision: ", precision(dev['labels'].tolist(), dev['predictions'].tolist()))
print("Recall: ", recall(dev['labels'].tolist(), dev['predictions'].tolist()))
print("F1: ", f1(dev['labels'].tolist(), dev['predictions'].tolist()))

tn, fp, fn, tp = confusion_matrix_values(dev['labels'].tolist(), dev['predictions'].tolist())
print("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))


converted_test_predictions = decode(test_predictions)

with open(os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE), 'w') as f:
    for item in converted_test_predictions:
        f.write("%s\n" % item)

