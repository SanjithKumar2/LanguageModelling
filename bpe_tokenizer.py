import dataclasses
import unicodedata
import re
import regex
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, Manager
from joblib import Parallel, delayed
import json
import logging
from utils import merge_tokens, get_stats_chunk
import argparse
from typing import Dict, Tuple, List
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class BPETokenizer:
    language: str
    vocab_size: int = 40000
    special_tokens: Dict[str, int] = dataclasses.field(default_factory=lambda: {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3})
    merges: Dict[Tuple[str, str], str] = dataclasses.field(default_factory=lambda: {})
    vocab_to_idx: Dict[str, int] = dataclasses.field(default_factory=lambda: {})
    idx_to_vocab: Dict[int, str] = dataclasses.field(default_factory=lambda: {})

    def __post_init__(self):
        if self.language not in ["ta", "en"]:
            raise ValueError("Only English and Tamil are supported")

    def normalize_text(self, text):
        if self.language == "ta":
            text = unicodedata.normalize('NFC', text)
            text = re.sub(r'"{2,}', '"', text)
            text = re.sub(r"'{2,}", "'", text)
            text = re.sub(r"[a-zA-Z]", '', text)
            text = re.sub(r'([.;!:?"\'])', r' \1 ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        elif self.language == "en":
            text = unicodedata.normalize('NFC', text)
            text = ''.join(c for c in text if (ord(c) >= 32 and ord(c) <= 126) or c in ["<PAD>", "<UNK>", "<BOS>", "<EOS>", " "])
            text = text.lower()
            text = re.sub(r'([.;!:?"\'])', r' \1 ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        else:
            raise ValueError("Unsupported language")
        
    def text_tokenize(self, text):
        text = self.normalize_text(text)
        text = text.split()
        
        words = []
        for word in text:
            graphemes = regex.findall(r'\X', word)
            words.append(graphemes + ["</w>"])
        
        return words
    @staticmethod
    def _get_stats_chunk(words_chunk):
        pairs = {}
        for word in words_chunk:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    @staticmethod
    def _merge_tokens(merge_info):
        words, new_tokens, best_pair = merge_info[0], merge_info[1], merge_info[2]
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(new_tokens)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words
                
    @staticmethod
    def _train_chunk(chunk_info):
        chunk, shared_merges, shared_vocab_to_idx, num_merges = chunk_info[0], chunk_info[1], chunk_info[2], chunk_info[3]
        local_merges = {pair : merged for pair, merged in shared_merges.items()}
        local_vocab = set()
        words = chunk[0]

        for _ in range(num_merges):
            current_pairs = {}
            for word in words:
                i = 0
                while i < len(word) - 1:
                    current_pairs[(word[i],word[i+1])] = current_pairs.get((word[i],word[i+1]),0) + 1
                    i+=1
            if not current_pairs:
                break
            
            best_pair = max(current_pairs.items(), key = lambda x: x[1])[0]
            new_token = best_pair[0] + best_pair[1]
            if new_token not in local_vocab:
                local_vocab.add(new_token)

            local_merges[best_pair] = new_token

            new_words = []

            for word in words:
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and word[i] == best_pair[0] and word[i+1] ==best_pair[1]:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_words.append(new_word)
            words = new_words


        with shared_merges._mutex:
            for pair, merge in local_merges.items():
                if pair not in shared_merges:
                    shared_merges[pair] = merge
        
        with shared_vocab_to_idx._mutex:
            for vocab in local_vocab:
                if vocab not in shared_vocab_to_idx:
                    shared_vocab_to_idx[vocab] = len(shared_vocab_to_idx)
        
        return words

    def train_bpe(self, text, output_file, num_merges=None, is_save=True, num_workers=None, use_multiprocess = False, use_prallel = False):
        words = self.text_tokenize(text)
        chars = set()
        for word in words:
            chars.update(word)
        for key, val in self.special_tokens.items():
            self.vocab_to_idx[key] = val
        for char in chars:
            self.vocab_to_idx[char] = len(self.vocab_to_idx)
        if num_merges is None:
                num_merges = self.vocab_size - len(chars) - len(self.special_tokens)
        if num_workers == None:
            

            for i in tqdm(range(num_merges)):

                pairs = BPETokenizer._get_stats_chunk(words)

                if not pairs:
                    logger.info("All Pairs were merged, stopping training.")
                    break

                best_pair = max(pairs.items(), key=lambda x: x[1])[0]

                new_token = best_pair[0] + best_pair[1]
                self.vocab_to_idx[new_token] = len(self.vocab_to_idx)

                self.merges[best_pair] = new_token

                merge_info = (words, new_token, best_pair)
                words = BPETokenizer._merge_tokens(merge_info)

                #if (i + 1) % 1000 == 0:
                #   logger.info(f"Completed {i + 1} merges. Current vocab size: {len(self.vocab_to_idx)}")

            logger.info(f"Done merges. Current vocab size: {len(self.vocab_to_idx)}")
            if is_save:
                self.save(output_file)
            return words
    
        elif use_multiprocess:
            manager = multiprocessing.Manager()
            shared_merges = manager.dict(self.merges)
            shared_vocab_to_idx = manager.dict(self.vocab_to_idx)
            shared_vocab_size = manager.Value('i', len(self.vocab_to_idx))
            pool = None
            try:
                cores = min(multiprocessing.cpu_count(), num_workers)
                chunk_size = len(words) // cores
                chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
                args_list = [(chunk, shared_merges, shared_vocab_to_idx, num_merges) for chunk in chunks]
                with multiprocessing.Pool(processes=cores) as pool:
                    results = pool.map(
                        BPETokenizer._train_chunk, args_list
                    )
                words = []
                for result in results:
                    words.extend(result)
                self.merges = dict(shared_merges)
                self.vocab_to_idx = dict(shared_vocab_to_idx)
                self.idx_to_vocab = {v: k for k, v in self.vocab_to_idx.items()}
                if is_save:
                    self.save(output_file)
                return words
                
            except KeyboardInterrupt:
                if pool is not None:
                    pool.terminate()
                raise
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
        elif use_prallel:
            with Parallel(n_jobs=num_workers, prefer="processes") as parallel:
                for _ in range(num_merges):
                    chunk_size = max(1, len(words) // multiprocessing.cpu_count())
                    chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]
                    
                    pair_counts = parallel(
                        delayed(self._get_stats_chunk)(chunk)
                        for chunk in chunks
                    )
                    pairs = defaultdict(int)
                    for chunk in pair_counts:
                        for pair, count in chunk.items():
                            pairs[pair] += count

                    if not pairs:
                        logger.info("All pairs merged, stopping early")
                        break

                    best_pair = max(pairs.items(), key=lambda x: x[1])[0]
                    new_token = "".join(best_pair)
                    self.merges[best_pair] = new_token
                    self.vocab_to_idx[new_token] = len(self.vocab_to_idx)
                    words = parallel(
                        delayed(self._merge_tokens)(chunk, best_pair, new_token)
                        for chunk in chunks
                    )
                    words = [word for chunk in words for word in chunk]

                 #   if (len(self.vocab_to_idx) % 1000) == 0:
                 #       logger.info(f"Merges completed: {len(self.merges)}, Vocab size: {len(self.vocab_to_idx)}")

            self.idx_to_vocab = {v: k for k, v in self.vocab_to_idx.items()}
            if is_save:
                self.save(output_file)
            return words
        else:
            pool = None
            try:
                cores = num_workers if num_workers else multiprocessing.cpu_count()
                chunk_size = len(words) // cores
                chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

                manager = Manager()
                shared_merges = manager.dict(self.merges)
                shared_vocab_to_idx = manager.dict(self.vocab_to_idx)

                with Pool(processes=cores) as pool:
                    for i in tqdm(range(num_merges)):
                        pair_counts = pool.map(get_stats_chunk, chunks)

                        global_pairs = defaultdict(int)
                        for count in pair_counts:
                            for pair, freq in count.items():
                                global_pairs[pair] += freq

                        if not global_pairs:
                            logger.info("All pairs merged, stopping training.")
                            break
                        best_pair = max(global_pairs.items(), key=lambda x: x[1])[0]
                        new_token = best_pair[0] + best_pair[1]

                        with shared_vocab_to_idx._mutex:
                            shared_vocab_to_idx[new_token] = len(shared_vocab_to_idx)
                        with shared_merges._mutex:
                            shared_merges[best_pair] = new_token
                        merge_args = [(chunk, new_token, best_pair) for chunk in chunks]
                        chunks = pool.map(merge_tokens, merge_args)

                  #      if (i + 1) % 1000 == 0:
                  #           logger.info(f"Completed {i + 1} merges. Current vocab size: {len(shared_vocab_to_idx)}")

                self.merges = dict(shared_merges)
                self.vocab_to_idx = dict(shared_vocab_to_idx)
                self.idx_to_vocab = {v: k for k, v in self.vocab_to_idx.items()}

                if is_save:
                    self.save(output_file)
                return words
            except KeyboardInterrupt:
                if pool is not None:
                    pool.terminate()
                raise
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()
        
    def tokenize(self, text):
        words = self.text_tokenize(text)
        tokenized = []
        for word_tokens in words:
            while len(word_tokens) > 1:
                best_pair = None
                best_idx = -1
                
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    if pair in self.merges:
                        best_pair = pair
                        best_idx = i
                        break
                
                if best_pair is None:
                    break
                    
                word_tokens[best_idx] = self.merges[best_pair]
                del word_tokens[best_idx + 1]
            
            for subword in word_tokens:
                if subword in self.vocab_to_idx:
                    tokenized.append(self.vocab_to_idx[subword])
                else:
                    tokenized.append(self.vocab_to_idx["<UNK>"])
        return tokenized
    
    def decode(self, tokens):
        text = [self.idx_to_vocab.get(i, "<UNK>") for i in tokens]
        text = "".join(text).replace("</w>", " ")
        if text.endswith(" "):
            text = text[:-1]
        return text
            
            
    def save(self, output_path) -> None:
        try:
            with open(f"{output_path}.merges.json", "w", encoding="utf-8") as f:
                serializable_merges = {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}
                json.dump(serializable_merges, f, ensure_ascii=False, indent=2)

            with open(f"{output_path}.vocab.json", 'w', encoding='utf-8') as f:
                json.dump(self.vocab_to_idx, f, ensure_ascii=False, indent=2)

            logger.info("Successfully saved tokenizer")
        except Exception as e:
            logger.info(f"save failed: {e}")

    @classmethod
    def load(cls, output_path, language) -> 'BPETokenizer':
        tokenizer = cls(language)

        with open(f"{output_path}.vocab.json", 'r', encoding='utf-8') as f:
            tokenizer.vocab_to_idx = json.load(f)
            tokenizer.vocab_to_idx = {k: int(v) if isinstance(v, str) and v.isdigit() else v 
                                  for k, v in tokenizer.vocab_to_idx.items()}
        with open(f"{output_path}.merges.json", 'r', encoding='utf-8') as f:
            serialized_merges = json.load(f)
            tokenizer.merges = {tuple(k.split()): v for k, v in serialized_merges.items()}
            
        tokenizer.idx_to_vocab = {v: k for k, v in tokenizer.vocab_to_idx.items()}
        
        return tokenizer

def learn_bpe(datasets=["full_corpa"], vocab_size={"source_lang": 20000, "target_lang": 20000}, 
              languages={"src": "ta", "trg": "en"}, output_prefixes={"src": "src_bpe1", "trg": "trg_bpe1"},num_workers=None):
    """
    Train BPE tokenizers for source and target languages on the specified datasets.
    """
    full_source_text = ""
    full_target_text = ""
    for dataset in datasets:
        with open(f"data_{dataset}/{dataset}_source_full.txt", 'r', encoding='utf-8') as f:
            full_source_text += " " + f.read()
        with open(f"data_{dataset}/{dataset}_target_full.txt", 'r', encoding='utf-8') as f:
            full_target_text += " " + f.read()
    
    full_src_vocab_size = vocab_size.get("source_lang", 20000)
    full_tgt_vocab_size = vocab_size.get("target_lang", 20000)
    
    # Train source language tokenizer (e.g., Tamil)
    src_tokenizer = BPETokenizer(languages["src"], full_src_vocab_size)
    src_tokenizer.train_bpe(full_source_text, output_prefixes["src"], num_workers=num_workers)
    logger.info(f"Trained BPE tokenizer for {languages['src']} and saved to {output_prefixes['src']}")
    
    # Train target language tokenizer (e.g., English)
    tgt_tokenizer = BPETokenizer(languages["trg"], full_tgt_vocab_size)
    tgt_tokenizer.train_bpe(full_target_text, output_prefixes["trg"], num_workers=num_workers)
    logger.info(f"Trained BPE tokenizer for {languages['trg']} and saved to {output_prefixes['trg']}")

def tokenize_text(input_file: str, output_file: str, language: str, tokenizer_path: str):
    """
    Tokenize the input text file using a pre-trained BPE tokenizer and save the tokenized output.
    """
    # Load the tokenizer
    tokenizer = BPETokenizer.load(tokenizer_path, language)
    
    # Read input text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize
    tokenized = tokenizer.tokenize(text)
    tokenized_ids = tokenized[0]  # Get the token IDs (integers)
    
    # Save tokenized output as a list of integers (one per line for readability)
    with open(output_file, 'w', encoding='utf-8') as f:
        for token_id in tokenized_ids:
            f.write(str(token_id) + '\n')
    
    logger.info(f"Tokenized {input_file} and saved to {output_file}")

def decode_text(input_file: str, output_file: str, language: str, tokenizer_path: str):
    """
    Decode the tokenized input file using a pre-trained BPE tokenizer and save the decoded text.
    """
    # Load the tokenizer
    tokenizer = BPETokenizer.load(tokenizer_path, language)
    
    # Read tokenized IDs (assuming one integer per line)
    tokenized_ids = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokenized_ids.append(int(line.strip()))
    
    # Decode
    decoded_text = tokenizer.decode(tokenized_ids)
    
    # Save decoded text
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(decoded_text)
    
    logger.info(f"Decoded {input_file} and saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BPE Tokenizer Script for Training, Tokenizing, and Decoding')
    
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'tokenize', 'decode'],
                        help='Mode of operation: train, tokenize, or decode')
    
    # Common arguments
    parser.add_argument('--language', type=str, help='Language for tokenizing/decoding (ta or en)')
    parser.add_argument('--tokenizer_path', type=str, help='Path to load/save the tokenizer (e.g., src_bpe1)')
    
    # Training-specific arguments
    parser.add_argument('--datasets', nargs='+', default=['full_corpa'],
                        help='List of datasets to train on (e.g., full_corpa)')
    parser.add_argument('--src_lang', type=str, default='ta', help='Source language (e.g., ta for Tamil)')
    parser.add_argument('--tgt_lang', type=str, default='en', help='Target language (e.g., en for English)')
    parser.add_argument('--src_vocab', type=int, default=20000, help='Vocabulary size for source language')
    parser.add_argument('--tgt_vocab', type=int, default=20000, help='Vocabulary size for target language')
    parser.add_argument('--output_prefixes', nargs=2, default=['src_bpe1', 'trg_bpe1'],
                        help='Output prefixes for source and target tokenizers (e.g., src_bpe1 trg_bpe1)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of wokers for parallel processing')
    
    # Tokenize/Decode-specific arguments
    parser.add_argument('--input_file', type=str, help='Input file for tokenizing or decoding')
    parser.add_argument('--output_file', type=str, help='Output file for tokenized or decoded text')

    args = parser.parse_args()

    if args.mode == 'train':
        vocab_size = {"source_lang": args.src_vocab, "target_lang": args.tgt_vocab}
        languages = {"src": args.src_lang, "trg": args.tgt_lang}
        output_prefixes = {"src": args.output_prefixes[0], "trg": args.output_prefixes[1]}
        learn_bpe(datasets=args.datasets, vocab_size=vocab_size, languages=languages, output_prefixes=output_prefixes, num_workers = args.num_workers)
    
    elif args.mode == 'tokenize':
        if not all([args.input_file, args.output_file, args.language, args.tokenizer_path]):
            parser.error("Tokenize mode requires --input_file, --output_file, --language, and --tokenizer_path")
        tokenize_text(args.input_file, args.output_file, args.language, args.tokenizer_path)
    
    elif args.mode == 'decode':
        if not all([args.input_file, args.output_file, args.language, args.tokenizer_path]):
            parser.error("Decode mode requires --input_file, --output_file, --language, and --tokenizer_path")
        decode_text(args.input_file, args.output_file, args.language, args.tokenizer_path)
