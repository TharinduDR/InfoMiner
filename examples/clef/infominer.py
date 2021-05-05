import shutil

import pandas as pd
import numpy as np
import os

import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


from examples.nlp4if.common.evaluation import precision, recall, f1, confusion_matrix_values
from examples.clef.infominer_config import TEMP_DIRECTORY, config, MODEL_TYPE, MODEL_NAME, SEED, \
    SUBMISSION_FILE, DEV_RESULT_FILE
from infominer.classification import ClassificationModel


if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)


languages = {
    "Arabic": ["examples/clef/data/CT21-AR-Train-T1-Labels.tsv",
              "examples/clef/data/CT21-AR-Dev-T1-Labels.tsv",
              "examples/clef/data/CT21-AR-Test-T1.tsv"],

    "Bulgarian": ["examples/clef/data/dataset_train_v1_bulgarian.tsv",
              "examples/clef/data/dataset_dev_v1_bulgarian.tsv",
              "examples/clef/data/dataset_test_input_bulgarian.tsv"],

    "English": ["examples/clef/data/dataset_train_v1_english.tsv",
              "examples/clef/data/dataset_dev_v1_english.tsv",
              "examples/clef/data/dataset_test_input_english.tsv"],

    "Spanish": ["examples/clef/data/dataset_train_spanish.tsv",
              "examples/clef/data/dataset_dev_spanish.tsv",
              "examples/clef/data/dataset_test_spanish.tsv"],

    "Turkish": ["examples/clef/data/dataset_train_v1_turkish.tsv",
              "examples/clef/data/dataset_test_v1_turkish.tsv",
              "examples/clef/data/dataset_dev_v1_turkish.tsv"],
}

train_list = []
dev_list = []
test_list = []

dev_sentences_list = []
test_sentences_list = []

for key, value in languages.items():
    print("Reading ", key)

    train_temp = pd.read_csv(value[0], sep='\t')
    dev_temp = pd.read_csv(value[1], sep='\t')
    test_temp = pd.read_csv(value[2], sep='\t')

    train_temp = train_temp[['tweet_text', 'claim_worthiness']]
    # dev_temp = dev_temp[['tweet_text', 'claim_worthiness']]
    # test_temp = test_temp[['tweet_text']]

    train_temp = train_temp.rename(columns={'tweet_text': 'text', 'claim_worthiness': 'labels'}).dropna()
    dev_temp = dev_temp.rename(columns={'tweet_text': 'text', 'claim_worthiness': 'labels'}).dropna()
    test_temp = test_temp.rename(columns={'tweet_text': 'text'}).dropna()

    dev_sentences_temp = dev_temp['text'].tolist()
    test_sentences_temp = test_temp['text'].tolist()

    train_list.append(train_temp)
    dev_list.append(dev_temp)
    test_list.append(test_temp)
    dev_sentences_list.append(dev_sentences_temp)
    test_sentences_list.append(test_sentences_temp)


train = pd.concat(train_list)
train = train.sample(frac=1)

dev_preds_list = []
test_preds_list = []

for dev, test in zip(dev_list, test_list):
    dev_preds = np.zeros((len(dev), config["n_fold"]))
    test_preds = np.zeros((len(test), config["n_fold"]))

    dev_preds_list.append(dev_preds)
    test_preds_list.append(test_preds)

for i in range(config["n_fold"]):
    if os.path.exists(config['output_dir']) and os.path.isdir(config['output_dir']):
        shutil.rmtree(config['output_dir'])
    print("Started Fold {}".format(i))
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=config, num_labels=1,
                                use_cuda=torch.cuda.is_available())
    train_df, eval_df = train_test_split(train, test_size=0.2, random_state=SEED * i)
    model.train_model(train_df, eval_df=eval_df, mae=mean_absolute_error)
    model = ClassificationModel(MODEL_TYPE, config["best_model_dir"], args=config, num_labels=1,
                                use_cuda=torch.cuda.is_available())

    for dev_sentences, test_sentences, dev_preds, test_preds in zip(dev_sentences_list, test_sentences_list, dev_preds_list,
                                                               test_preds_list):
        dev_predictions, dev_raw_outputs = model.predict(dev_sentences)
        test_predictions, test_raw_outputs = model.predict(test_sentences)
        dev_preds[:, i] = dev_predictions
        test_preds[:, i] = test_predictions

for dev, dev_preds, test, test_preds in zip(dev_list, dev_preds_list, test_list, test_preds_list):
    dev['predictions'] = dev_preds.mean(axis=1)
    test['predictions'] = test_preds.mean(axis=1)


for dev, test, language in zip(dev_list, test_list,  [*languages]):
    dev["run_id"] = MODEL_NAME
    test["run_id"] = MODEL_NAME

    dev = dev[['topic_id', 'tweet_id', "predictions", "run_id"]]
    test = test[['topic_id', 'tweet_id', "predictions", "run_id"]]
    dev.to_csv(os.path.join(TEMP_DIRECTORY, DEV_RESULT_FILE.split(".")[0] + "_" + language + "." + DEV_RESULT_FILE.split(".")[1]), sep='\t', index=False, header=False)
    test.to_csv(os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE.split(".")[0] + "_" + language + "." + SUBMISSION_FILE.split(".")[1]), sep='\t', index=False, header=False)






