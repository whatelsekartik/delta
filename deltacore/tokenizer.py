# deltacore/tokenizer.py
"""
Custom Byte-Level Byte-Pair Encoding (BPE) Tokenizer implemented from scratch.
Audited to enforce strict Split-Before-Tokenization provenance:
- Records SHA-256 hash and byte length of the training split used to learn merges.
- Zero external tokenizer dependencies.
"""

import hashlib
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple


class ByteBPETokenizer:
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    NUM_SPECIAL = 4

    def __init__(self) -> None:
        self.merges: List[Tuple[int, int, int]] = []  # (id_a, id_b, new_id)
        self.merge_map: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, List[int]] = {}
        self.trained_on_split: str = "untrained"
        self.train_corpus_sha256: str = ""
        self.train_corpus_bytes: int = 0
        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        self.vocab = {
            self.PAD_ID: [],
            self.BOS_ID: [],
            self.EOS_ID: [],
            self.UNK_ID: [],
        }
        for b in range(256):
            self.vocab[self.NUM_SPECIAL + b] = [b]
        self.merges = []
        self.merge_map = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(
        self,
        train_text: str,
        target_vocab_size: int = 512,
        split_name: str = "train_only",
    ) -> None:
        """
        Trains BPE merges exclusively on the provided training split text.
        Records cryptographic provenance so the audit can verify no val/test leakage.
        """
        if split_name != "train_only":
            raise ValueError("Tokenizer MUST be trained exclusively on 'train_only' split.")

        self._init_base_vocab()
        raw_bytes = train_text.encode("utf-8")
        self.trained_on_split = split_name
        self.train_corpus_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        self.train_corpus_bytes = len(raw_bytes)

        chunks: List[List[int]] = []
        current: List[int] = []
        for b in raw_bytes:
            current.append(b + self.NUM_SPECIAL)
            if b in (32, 10, 13, 9, 46, 44, 58, 59) and len(current) >= 1:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        chunk_counts = Counter(tuple(c) for c in chunks)
        work_chunks: List[Tuple[List[int], int]] = [
            (list(k), v) for k, v in chunk_counts.items()
        ]

        next_id = self.NUM_SPECIAL + 256
        while next_id < target_vocab_size:
            pair_counts: Counter[Tuple[int, int]] = Counter()
            for seq, freq in work_chunks:
                if len(seq) < 2:
                    continue
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += freq

            if not pair_counts:
                break

            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < 2:
                break

            a, b = best_pair
            self.merges.append((a, b, next_id))
            self.merge_map[(a, b)] = next_id
            self.vocab[next_id] = self.vocab[a] + self.vocab[b]

            for idx, (seq, freq) in enumerate(work_chunks):
                if len(seq) < 2:
                    continue
                new_seq: List[int] = []
                i = 0
                n = len(seq)
                while i < n:
                    if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                        new_seq.append(next_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                work_chunks[idx] = (new_seq, freq)

            next_id += 1

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        raw_bytes = text.encode("utf-8")
        ids = [b + self.NUM_SPECIAL for b in raw_bytes]
        if not ids:
            return ([self.BOS_ID] if add_bos else []) + ([self.EOS_ID] if add_eos else [])

        for a, b, new_id in self.merges:
            if len(ids) < 2:
                break
            new_ids: List[int] = []
            i = 0
            n = len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, token_ids: List[int]) -> str:
        byte_list: List[int] = []
        for tid in token_ids:
            if tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID):
                continue
            byte_list.extend(self.vocab.get(int(tid), []))
        return bytes(byte_list).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "trained_on_split": self.trained_on_split,
            "train_corpus_sha256": self.train_corpus_sha256,
            "train_corpus_bytes": self.train_corpus_bytes,
            "merges": self.merges,
            "vocab": {str(k): v for k, v in self.vocab.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ByteBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls()
        tok.trained_on_split = payload.get("trained_on_split", "unknown")
        tok.train_corpus_sha256 = payload.get("train_corpus_sha256", "")
        tok.train_corpus_bytes = int(payload.get("train_corpus_bytes", 0))
        tok.merges = [tuple(m) for m in payload["merges"]]
        tok.merge_map = {(m[0], m[1]): m[2] for m in tok.merges}
        tok.vocab = {int(k): v for k, v in payload["vocab"].items()}
        return tok
