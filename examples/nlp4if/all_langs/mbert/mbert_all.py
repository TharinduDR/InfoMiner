import shutil

import pandas as pd
import numpy as np
import os

import torch
from sklearn.model_selection import train_test_split

from examples.nlp4if.common.converter import encode, decode
from examples.nlp4if.common.evaluation import precision, recall, f1, confusion_matrix_values
from examples.nlp4if.all_langs.mbert.mbert_all_config import TEMP_DIRECTORY, config, MODEL_TYPE, MODEL_NAME, SEED, \
    SUBMISSION_FILE
from examples.sample_size_counter import sample_size_counter,data_balancer
from infominer.classification import ClassificationModel

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

languages = {
    "Arabic": ["examples/nlp4if/all_langs/data/arabic/covid19_disinfo_binary_arabic_train.tsv",
               "examples/nlp4if/all_langs/data/arabic/covid19_disinfo_binary_arabic_dev_input.tsv",
               "examples/nlp4if/all_langs/data/arabic/covid19_disinfo_binary_arabic_test_gold.tsv"],

    "Bulgarian": ["examples/nlp4if/all_langs/data/bulgarian/covid19_disinfo_binary_bulgarian_train.tsv",
                  "examples/nlp4if/all_langs/data/bulgarian/covid19_disinfo_binary_bulgarian_dev.tsv",
                  "examples/nlp4if/all_langs/data/bulgarian/covid19_disinfo_binary_bulgarian_test_gold.tsv"],

    "English": ["examples/nlp4if/all_langs/data/english/covid19_disinfo_binary_english_train.tsv",
                "examples/nlp4if/all_langs/data/english/covid19_disinfo_binary_english_dev_input.tsv",
                "examples/nlp4if/all_langs/data/english/covid19_disinfo_binary_english_test_gold.tsv"]
    #
    # "Spanish": ["examples/all_langs/data/dataset_train_spanish.tsv",
    #           "examples/all_langs/data/dataset_dev_spanish.tsv",
    #           "examples/all_langs/data/dataset_test_spanish.tsv"],
    #
    # "Turkish": ["examples/all_langs/data/dataset_train_v1_turkish.tsv",
    #           "examples/all_langs/data/dataset_test_v1_turkish.tsv",
    #           "examples/all_langs/data/dataset_dev_v1_turkish.tsv"],
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
    test_temp.dropna(subset=["q1_label"], inplace=True)

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

    test_temp['labels'] = encode(test_temp["q1_label"])
    test_temp = test_temp[['text', 'labels']]

    dev_sentences_temp = dev_temp['text'].tolist()
    test_sentences_temp = test_temp['text'].tolist()

    train_list.append(train_temp)
    dev_list.append(dev_temp)
    test_list.append(test_temp)

    dev_sentences_list.append(dev_sentences_temp)
    test_sentences_list.append(test_sentences_temp)

# train = pd.concat(train_list)
train = data_balancer(train_list)

ara_dev = dev_list[0]
bul_dev = dev_list[1]
eng_dev = dev_list[2]

ara_test = test_list[0]
bul_test = test_list[1]
eng_test = test_list[2]

# shuffle samples in train set
train = train.sample(frac=1)

ara_dev_preds = np.zeros((len(dev_sentences_list[0]), config["n_fold"]))
bul_dev_preds = np.zeros((len(dev_sentences_list[1]), config["n_fold"]))
eng_dev_preds = np.zeros((len(dev_sentences_list[2]), config["n_fold"]))

ara_test_preds = np.zeros((len(test_sentences_list[0]), config["n_fold"]))
bul_test_preds = np.zeros((len(test_sentences_list[1]), config["n_fold"]))
eng_test_preds = np.zeros((len(test_sentences_list[2]), config["n_fold"]))

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

    # predict Arabic dev
    ara_test_final_predictions, raw_outputs = model.predict(dev_sentences_list[0])
    ara_dev_preds[:, i] = ara_test_final_predictions

    # predict Bulgarian dev
    bul_test_final_predictions, raw_outputs = model.predict(dev_sentences_list[1])
    bul_dev_preds[:, i] = bul_test_final_predictions

    # predict English dev
    eng_test_final_predictions, raw_outputs = model.predict(dev_sentences_list[2])
    eng_dev_preds[:, i] = eng_test_final_predictions

    # predict Arabic test
    ara_test_predictions, raw_outputs = model.predict(test_sentences_list[0])
    ara_test_preds[:, i] = ara_test_predictions

    # predict Bulgarian test
    bul_test_predictions, raw_outputs = model.predict(test_sentences_list[1])
    bul_test_preds[:, i] = bul_test_predictions

    # predict English test
    eng_test_predictions, raw_outputs = model.predict(test_sentences_list[2])
    eng_test_preds[:, i] = eng_test_predictions

    # test_predictions, test_raw_outputs = model.predict(test_sentences_list)
    # test_preds[:, i] = test_predictions

    print("Completed Fold {}".format(i))

########################## DEV ##############################
# select majority class of each instance (row)
ara_dev_final_predictions = []
for row in ara_dev_preds:
    row = row.tolist()
    ara_dev_final_predictions.append(int(max(set(row), key=row.count)))

ara_dev["predictions"] = ara_dev_final_predictions

print("--------------Arabic--------------")
print("Precision: ", precision(ara_dev['labels'].tolist(), ara_dev['predictions'].tolist()))
print("Recall: ", recall(ara_dev['labels'].tolist(), ara_dev['predictions'].tolist()))
print("F1: ", f1(ara_dev['labels'].tolist(), ara_dev['predictions'].tolist()))

tn, fp, fn, tp = confusion_matrix_values(ara_dev['labels'].tolist(), ara_dev['predictions'].tolist())
print("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))

# select majority class of each instance (row)
bul_dev_final_predictions = []
for row in bul_dev_preds:
    row = row.tolist()
    bul_dev_final_predictions.append(int(max(set(row), key=row.count)))

bul_dev["predictions"] = bul_dev_final_predictions

print("--------------Bulgarian--------------")
print("Precision: ", precision(bul_dev['labels'].tolist(), bul_dev['predictions'].tolist()))
print("Recall: ", recall(bul_dev['labels'].tolist(), bul_dev['predictions'].tolist()))
print("F1: ", f1(bul_dev['labels'].tolist(), bul_dev['predictions'].tolist()))

tn, fp, fn, tp = confusion_matrix_values(bul_dev['labels'].tolist(), bul_dev['predictions'].tolist())
print("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))

# select majority class of each instance (row)
eng_dev_final_predictions = []
for row in eng_dev_preds:
    row = row.tolist()
    eng_dev_final_predictions.append(int(max(set(row), key=row.count)))

eng_dev["predictions"] = eng_dev_final_predictions

print("--------------English--------------")
print("Precision: ", precision(eng_dev['labels'].tolist(), eng_dev['predictions'].tolist()))
print("Recall: ", recall(eng_dev['labels'].tolist(), eng_dev['predictions'].tolist()))
print("F1: ", f1(eng_dev['labels'].tolist(), eng_dev['predictions'].tolist()))

tn, fp, fn, tp = confusion_matrix_values(eng_dev['labels'].tolist(), eng_dev['predictions'].tolist())
print("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))

########################## TEST #############################
# select majority class of each instance (row)
ara_test_final_predictions = []
for row in ara_test_preds:
    row = row.tolist()
    ara_test_final_predictions.append(int(max(set(row), key=row.count)))

ara_test["predictions"] = ara_test_final_predictions

print("--------------Arabic--------------")
print("Precision: ", precision(ara_test['labels'].tolist(), ara_test['predictions'].tolist()))
print("Recall: ", recall(ara_test['labels'].tolist(), ara_test['predictions'].tolist()))
print("F1: ", f1(ara_test['labels'].tolist(), ara_test['predictions'].tolist()))

tn, fp, fn, tp = confusion_matrix_values(ara_test['labels'].tolist(), ara_test['predictions'].tolist())
print("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))

# select majority class of each instance (row)
bul_test_final_predictions = []
for row in bul_test_preds:
    row = row.tolist()
    bul_test_final_predictions.append(int(max(set(row), key=row.count)))

bul_test["predictions"] = bul_test_final_predictions

print("--------------Bulgarian--------------")
print("Precision: ", precision(bul_test['labels'].tolist(), bul_test['predictions'].tolist()))
print("Recall: ", recall(bul_test['labels'].tolist(), bul_test['predictions'].tolist()))
print("F1: ", f1(bul_test['labels'].tolist(), bul_test['predictions'].tolist()))

tn, fp, fn, tp = confusion_matrix_values(bul_test['labels'].tolist(), bul_test['predictions'].tolist())
print("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))

# select majority class of each instance (row)
eng_test_final_predictions = []
for row in eng_test_preds:
    row = row.tolist()
    eng_test_final_predictions.append(int(max(set(row), key=row.count)))

eng_test["predictions"] = eng_test_final_predictions

print("--------------English--------------")
print("Precision: ", precision(eng_test['labels'].tolist(), eng_test['predictions'].tolist()))
print("Recall: ", recall(eng_test['labels'].tolist(), eng_test['predictions'].tolist()))
print("F1: ", f1(eng_test['labels'].tolist(), eng_test['predictions'].tolist()))

tn, fp, fn, tp = confusion_matrix_values(eng_test['labels'].tolist(), eng_test['predictions'].tolist())
print("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))

# select majority class of each instance (row)
# test_predictions = []
# for row in test_preds:
#     row = row.tolist()
#     test_predictions.append(int(max(set(row), key=row.count)))


# converted_test_predictions = decode(test_predictions)
#
# with open(os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE), 'w') as f:
#     for item in converted_test_predictions:
#         f.write("%s\n" % item)
