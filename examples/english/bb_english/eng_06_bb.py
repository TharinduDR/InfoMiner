import shutil

import pandas as pd
import numpy as np
import os

import torch
from sklearn.model_selection import train_test_split

from examples.common.converter import encode
from examples.common.evaluation import precision, recall, f1, confusion_matrix_values
from examples.english.bb_english.eng_bb_config import TEMP_DIRECTORY, config, MODEL_TYPE, MODEL_NAME, SEED
from infominer.classification import ClassificationModel


if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

full = pd.read_csv(os.path.join("examples", "english", "data", "covid19_disinfo_binary_english_train.tsv"), sep='\t')
full.dropna(subset=["q5_label"], inplace=True)
full['labels'] = encode(full["q6_label"])
full = full[['tweet_text', 'labels']]
full = full.rename(columns={'tweet_text': 'text'})


train, dev = train_test_split(full, test_size=0.1, random_state=777)

dev_sentences = dev['text'].tolist()
dev_preds = np.zeros((len(dev_sentences), config["n_fold"]))

for i in range(config["n_fold"]):
    if os.path.exists(config['output_dir']) and os.path.isdir(config['output_dir']):
        shutil.rmtree(config['output_dir'])
    print("Started Fold {}".format(i))
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=config,
                                use_cuda=torch.cuda.is_available())
    train_df, eval_df = train_test_split(full, test_size=0.1, random_state=SEED * i)
    model.train_model(train_df, eval_df=eval_df, precision=precision, recall=recall, f1=f1)
    model = ClassificationModel(MODEL_TYPE, config["best_model_dir"], args=config,
                                use_cuda=torch.cuda.is_available())

    predictions, raw_outputs = model.predict(dev_sentences)
    dev_preds[:, i] = predictions

    print("Completed Fold {}".format(i))

# select majority class of each instance (row)
dev_predictions = []
for row in dev_preds:
    row = row.tolist()
    dev_predictions.append(int(max(set(row), key=row.count)))


dev["predictions"] = dev_predictions


print("Precision: ", precision(dev['labels'].tolist(), dev['predictions'].tolist()))
print("Recall: ", recall(dev['labels'].tolist(), dev['predictions'].tolist()))
print("F1: ", f1(dev['labels'].tolist(), dev['predictions'].tolist()))

tn, fp, fn, tp = confusion_matrix_values(dev['labels'].tolist(), dev['predictions'].tolist())
print("Confusion Matrix (tn, fp, fn, tp) {} {} {} {}".format(tn, fp, fn, tp))



