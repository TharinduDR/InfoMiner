import shutil

import pandas as pd
import numpy as np
import os

import torch
from sklearn.model_selection import train_test_split

from examples.common.converter import encode, decode
from examples.task1a.evaluation import f1, precision, recall
from examples.task1a.infominer_config import MODEL_TYPE, MODEL_NAME, config, SEED, TEMP_DIRECTORY
from infominer.classification import ClassificationModel

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

full = pd.read_csv(os.path.join("examples", "task1a", "data", "train.tsv"), sep='\t', names=['id', 'text', "row_class"], header=None)
full['labels'] = encode(full["row_class"])
full = full[['text', 'labels']]


dev_df = pd.read_csv(os.path.join("examples", "task1a", "data", "tweets.tsv"), sep='\t')
dev_sentences = dev_df['tweet'].tolist()


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
final_predictions = []
for row in dev_preds:
    row = row.tolist()
    final_predictions.append(int(max(set(row), key=row.count)))


dev_submission = percentile_list = pd.DataFrame(
    {'tweet_id': dev_df["tweet_id"].tolist(),
     'label': final_predictions
    })

dev_labels = pd.read_csv(os.path.join("examples", "task1a", "data", "class.tsv"), sep='\t')
dev_labels["label"] = encode(dev_labels["label"])

print(precision(dev_labels["label"].tolist(), dev_submission['label'].tolist()))
print(recall(dev_labels["label"].tolist(), dev_submission['label'].tolist()))
print(f1(dev_labels["label"].tolist(), dev_submission['label'].tolist()))

dev_submission['label'] = decode(dev_submission['label'])
dev_submission.to_csv(os.path.join(TEMP_DIRECTORY, "dev_class.tsv"), sep='\t', encoding='utf-8', index=False)



