import json
import math
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from args import args


class PepDataset(Dataset):
    def __init__(
        self,
        path,
        mode="regression",
        threshold=16.0,
        max_length=None,
        tokenizer_path="Data/Vocab",
        vocab_path="Data/Vocab/voc.json",
    ):
        self.path = path
        self.mode = mode
        self.threshold = threshold
        self.tokenizer_path = tokenizer_path
        self.vocab_path = vocab_path

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.amino_acid_vocab = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, ignore_mismatched_sizes=True
        )

        self.sequences, self.raw_labels = self._read_file_csv(path)

        if max_length is None:
            self.max_length = self._calculate_max_length(self.sequences)
        else:
            if max_length <= 0:
                raise ValueError("max_length must be a positive integer.")
            self.max_length = max_length

        self.padded_sequences = self._pad_sequences(self.sequences, self.max_length)
        self.num_classes = 2

    def _read_file_csv(self, file_path):
        df = pd.read_csv(file_path)
        if "SEQUENCE" not in df.columns or "MIC" not in df.columns:
            raise ValueError(
                "CSV file must contain 'SEQUENCE' and 'MIC' columns."
            )

        sequences = df["SEQUENCE"].tolist()
        sequences = [str(seq).upper().strip() for seq in sequences]
        mic_values = df["MIC"].tolist()
        mic_values = [float(v) for v in mic_values]

        valid_indices = [
            i for i, (seq, mic) in enumerate(zip(sequences, mic_values))
            if len(seq) > 0 and mic > 0
        ]
        if len(valid_indices) < len(sequences):
            sequences = [sequences[i] for i in valid_indices]
            mic_values = [mic_values[i] for i in valid_indices]

        return sequences, mic_values

    def _calculate_max_length(self, sequences):
        if not sequences:
            raise ValueError("Sequence list is empty.")
        raw_max = max(len(seq) for seq in sequences)
        return raw_max + 2

    def _pad_sequences(self, sequences, max_length):
        padded_sequences = []
        for seq in sequences:
            tokens = [self.tokenizer.bos_token] + list(seq) + [self.tokenizer.eos_token]
            if len(tokens) < max_length:
                tokens += [self.tokenizer.pad_token] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
            padded_sequences.append(tokens)
        return padded_sequences

    def set_mode(self, mode):
        if mode not in ("regression", "classification"):
            raise ValueError("Mode must be 'regression' or 'classification'.")
        self.mode = mode

    def set_threshold(self, threshold):
        self.threshold = float(threshold)

    def __len__(self):
        return len(self.padded_sequences)

    def __getitem__(self, idx):
        tokens = self.padded_sequences[idx]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        pad_token_id = self.tokenizer.vocab.get(
            self.tokenizer.pad_token, self.tokenizer.pad_token_id
        )
        if isinstance(pad_token_id, str):
            pad_token_id = self.tokenizer.vocab.get(pad_token_id, 0)
        attention_mask = (input_ids != pad_token_id).long()

        raw_label = self.raw_labels[idx]

        if self.mode == "classification":
            label = 1 if raw_label >= self.threshold else 0
            label = torch.tensor(label, dtype=torch.long)
        elif self.mode == "regression":
            label = torch.tensor(
                -math.log10(raw_label / 1000000.0), dtype=torch.float32
            )
        else:
            raise ValueError(
                f"Unknown mode: {self.mode}. Supported modes: 'regression', 'classification'."
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }

    def __repr__(self):
        return (
            f"PepDataset(path='{self.path}', mode='{self.mode}', "
            f"threshold={self.threshold}, max_length={self.max_length}, "
            f"num_sequences={len(self)})"
        )


if __name__ == "__main__":
    dataset = PepDataset(
        path="Data/Dataset/E_coli.csv",
        mode=args.mode,
        threshold=args.threshold,
        max_length=args.max_len,
    )
    print(dataset)
    print("Max length:", dataset.max_length)
    sample = dataset[0]
    print("Sample input_ids shape:", sample["input_ids"].shape)
    print("Sample attention_mask shape:", sample["attention_mask"].shape)
    print("Sample label:", sample["label"])