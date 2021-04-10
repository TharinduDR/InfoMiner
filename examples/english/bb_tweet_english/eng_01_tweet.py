import shutil

import pandas as pd
import numpy as np
import os

import torch
from sklearn.model_selection import train_test_split

from examples.common.converter import encode, decode
from examples.common.evaluation import precision, recall, f1, confusion_matrix_values
from examples.english.bb_tweet_english.eng_tweet_config import TEMP_DIRECTORY, config, MODEL_TYPE, MODEL_NAME, SEED, \
    SUBMISSION_FILE
from infominer.classification import ClassificationModel


if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

train = pd.read_csv(os.path.join("examples", "english", "data", "covid19_disinfo_binary_english_train.tsv"), sep='\t')
train['labels'] = encode(train["q1_label"])
train = train[['tweet_text', 'labels']]
train = train.rename(columns={'tweet_text': 'text'})


dev = pd.read_csv(os.path.join("examples", "english", "data", "covid19_disinfo_binary_english_dev_input.tsv"), sep='\t')
dev['labels'] = encode(dev["q1_label"])
dev = dev[['tweet_text', 'labels']]
dev = dev.rename(columns={'tweet_text': 'text'})

test = pd.read_csv(os.path.join("examples", "english", "data", "covid19_disinfo_binary_english_test_input.tsv"), sep='\t')

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



