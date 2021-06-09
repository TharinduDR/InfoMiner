import shutil

import pandas as pd
import numpy as np
import os

import torch
from sklearn.model_selection import train_test_split

from examples.nlp4if.common.converter import encode, decode
from examples.nlp4if.common.evaluation import precision, recall, f1, confusion_matrix_values
from examples.nlp4if.clef.mbert.clef_mbert_config import TEMP_DIRECTORY, config, MODEL_TYPE, MODEL_NAME, SEED, \
    SUBMISSION_FILE
from examples.sample_size_counter import sample_size_counter
from infominer.classification import ClassificationModel

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

languages = {
    "Arabic": ["examples/nlp4if/clef/data/arabic/covid19_disinfo_binary_arabic_train.tsv",
               "examples/nlp4if/clef/data/arabic/covid19_disinfo_binary_arabic_dev_input.tsv",
               "examples/nlp4if/clef/data/arabic/covid19_disinfo_binary_arabic_test_input.tsv"],

    "Bulgarian": ["examples/nlp4if/clef/data/bulgarian/covid19_disinfo_binary_bulgarian_train.tsv",
                  "examples/nlp4if/clef/data/bulgarian/covid19_disinfo_binary_bulgarian_dev_input.tsv",
                  "examples/nlp4if/clef/data/bulgarian/covid19_disinfo_binary_bulgarian_test_input.tsv"],

    "English": ["examples/nlp4if/clef/data/english/covid19_disinfo_binary_english_train.tsv",
                "examples/nlp4if/clef/data/english/covid19_disinfo_binary_english_dev_input.tsv",
                "examples/nlp4if/clef/data/english/covid19_disinfo_binary_english_test_input.tsv"]
    #
    # "Spanish": ["examples/clef/data/dataset_train_spanish.tsv",
    #           "examples/clef/data/dataset_dev_spanish.tsv",
    #           "examples/clef/data/dataset_test_spanish.tsv"],
    #
    # "Turkish": ["examples/clef/data/dataset_train_v1_turkish.tsv",
    #           "examples/clef/data/dataset_test_v1_turkish.tsv",
    #           "examples/clef/data/dataset_dev_v1_turkish.tsv"],
}

train_list = []
dev_list = []
test_list = []

dev_sentences_list = []
test_sentences_list = []

for key, value in languages.items():
    print("Reading: ", key)

    train_temp = pd.read_csv(value[0], sep='\t')
    dev_temp = pd.read_csv(value[1], sep='\t')
    test_temp = pd.read_csv(value[2], sep='\t')

    train_temp.dropna(subset=["q1_label"], inplace=True)
    dev_temp.dropna(subset=["q1_label"], inplace=True)

    # Class count
    count_class_no, count_class_yes = train_temp.q1_label.value_counts().sort_index(ascending=True)

    # Divide by class
    df_class_no = train_temp[train_temp['q1_label'] == "no"]
    df_class_yes = train_temp[train_temp['q1_label'] == "yes"]

    size_counter = sample_size_counter(count_class_no, count_class_yes, 1)
    print("NOs : ", df_class_no['q1_label'].count())
    print("YESs : ", df_class_yes['q1_label'].count())
    print("size counter : ", size_counter)

    if size_counter > 0:
        df_class_no_under = df_class_no.sample(count_class_yes * size_counter)
        print("under sized NOs : ", df_class_no_under['q1_label'].count())
        train_temp = pd.concat([df_class_no_under, df_class_yes], axis=0)

    else:

        df_class_yes_under = df_class_yes.sample(count_class_no * abs(size_counter))
        print("under sized YESs : ", df_class_yes_under['q1_label'].count())
        train_temp = pd.concat([df_class_yes_under, df_class_no], axis=0)

    train_temp['labels'] = encode(train_temp["q1_label"])
    train_temp = train_temp[['text', 'labels']]

    dev_temp['labels'] = encode(dev_temp["q1_label"])
    dev_temp = dev_temp[['text', 'labels']]

    dev_sentences_temp = dev_temp['text'].tolist()
    test_sentences_temp = test_temp['text'].tolist()

    train_list.append(train_temp)
    dev_list.append(dev_temp)
    test_list.append(test_temp)
    dev_sentences_list.append(dev_sentences_temp)
    test_sentences_list.append(test_sentences_temp)

train = pd.concat(train_list)
dev = pd.concat(dev_list)
test = pd.concat(test_list)

# shuffle samples in train set
train = train.sample(frac=1)

dev_preds = np.zeros((len(dev_sentences_list), config["n_fold"]))
test_preds = np.zeros((len(test_sentences_list), config["n_fold"]))

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

    predictions, raw_outputs = model.predict(dev_sentences_list)
    dev_preds[:, i] = predictions

    test_predictions, test_raw_outputs = model.predict(test_sentences_list)
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
