import logging

import pandas as pd
from nlgeval import getListRouge, compute_metrics
from transformers import T5Tokenizer
from t5_model import T5Model

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = pd.read_csv('data/train.csv')
eval_df = pd.read_csv('data/valid.csv')
test_df = pd.read_csv('data/test.csv')

train_df.columns = ["prefix", "input_text", "target_text"]
eval_df.columns = ["prefix", "input_text", "target_text"]
test_df.columns = ["prefix", "input_text", "target_text"]


model_args = {
    "overwrite_output_dir": True,
    "fp16":True,
    "train_batch_size": 16,
    "num_train_epochs": 30,
    "max_seq_length":256,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": True,
    "evaluate_during_training_steps": 1000,
    "evaluate_generated_text": True,
    "evaluate_during_training": True,
    "evaluate_duringx_training_verbose": True,
    "use_multiprocessing": False,
    "use_multiprocessing_for_evaluation":False,
    "save_best_model": True,
    "top_k":5,
    "top_p":0.95,
    "max_length": 30,
    "use_early_stopping":True,
    "length_penalty":1.2,
    "best_model_dir":'result/best_model',
    "output_dir":'result',
    "save_steps":1000,
    "early_stopping_metric": 'Rouge_L',
    "early_stopping_metric_minimize": False,
}

from nlgeval import getListRouge, compute_metrics, _strip, Rouge

def getListRouge(hyp_list, refs):
    ref_list = []
    ref_list.append(refs)
    ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    assert len(refs) == len(hyps)
    ret_scores = {}
    scorers = [
        (Rouge(), "ROUGE_L")
    ]
    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hyps)
        if isinstance(method, list):
            for sc, scs, m in zip(score, scores, method):
                ret_scores[m] = sc
        else:
            ret_scores[method] = score
    del scorers
    return ret_scores['ROUGE_L']


def Rouge_L(labels, preds):
    score = getListRouge(preds, labels)
    return score

model = T5Model(model_type='t5', model_name="t5-base", args=model_args, tokenizer=None)
#
model.train_model(train_df, eval_data=eval_df, Rouge_L=Rouge_L)
