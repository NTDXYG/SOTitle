import logging
import os
import pickle

from torch.utils.data import Dataset
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def preprocess_data(data):
    prefix, input_text, target_text, tokenizer, args = data

    # Add EOS again if truncated?
    if args.preprocess_inputs:
        batch = tokenizer.prepare_seq2seq_batch(
            src_texts=[prefix + ": " + input_text],
            tgt_texts=[target_text],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

    else:
        batch = tokenizer.prepare_seq2seq_batch(
            src_texts=[prefix + input_text],
            tgt_texts=[target_text],
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

    input_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]
    return (input_ids, attention_mask, labels)


class T5Dataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir,
            mode + "_cached_"
            + str(args.max_seq_length)
            + str(len(data)),
        )
        print(cached_features_file)
        if os.path.exists(cached_features_file):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            data = [
                (str(prefix), str(input_text), str(target_text), tokenizer, args)
                for prefix, input_text, target_text in zip(
                    data["prefix"], data["input_text"], data["target_text"]
                )
            ]

            self.examples = [
                preprocess_data(d) for d in tqdm(data, disable=args.silent)
            ]

            logger.info(
                " Saving features into cached file %s", cached_features_file
            )
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
