from collections import defaultdict
import os
import pandas as pd
import logging

logger = logging.getLogger("TA-EN NMT")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_stats_chunk(words_chunk):
    pairs = defaultdict(int)
    for word in words_chunk:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += 1
    return pairs

def merge_tokens(args):
    chunk, new_token, best_pair = args
    new_words = []
    for word in chunk:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_words.append(new_word)
    return new_words

def prepare_data(csv_path = "", train_size=0.9, no_of_samples = 20000, df = None, name = None, shuffle = True):
    try:
        if csv_path:
            df = pd.read_csv(csv_path)
        if not name:
            name = os.path.basename(csv_path).split(".")[0]
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.rename(columns={"en": "English", "ta": "Tamil"})
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        df = df.iloc[:no_of_samples]
        folder = f"./data_{name.strip()}/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        df["Tamil"].to_csv(f"{folder}//{name}_source_full.txt", index=False, header=False)
        df["English"].to_csv(f"{folder}//{name}_target_full.txt", index=False, header=False)
        
        train_size = int(train_size * len(df))
        train_data = df.iloc[:train_size]
        valid_data = df.iloc[train_size:]

        train_data["Tamil"].to_csv(f"{folder}//{name}_source_train.txt", index=False, header=False)
        train_data["English"].to_csv(f"{folder}//{name}_target_train.txt", index=False, header=False)
        valid_data["Tamil"].to_csv(f"{folder}//{name}_source_valid.txt", index=False, header=False)
        valid_data["English"].to_csv(f"{folder}//{name}_target_valid.txt", index=False, header=False)
        return True
    except Exception as e:
        logger.error(f"UNABLE TO PREPARE DATA:{e}")
        return False
    
def combine_datasets(root_dir,dataset_path = None, no_of_samples=2000,train_size = 0.9, excluded = ["corpus.bcn.test 2k.csv","corpus.bcn.dev 1k.csv","parallel 8k gloss.xlsx"], condense=False):
    try:
        files = [file for file in os.listdir(root_dir) if file not in excluded]
        if dataset_path:
            assert os.path.basename(dataset_path) in files, f"Dataset {dataset_path} not found in {root_dir}"
            if not prepare_data(os.path.join(root_dir,dataset_path),train_size=train_size,no_of_samples=no_of_samples):
                return False
        elif condense:
            base_file = pd.read_csv(os.path.join(root_dir,files[0]))
            for file in files[1:]:
                base_file = pd.concat([base_file,pd.read_csv(os.path.join(root_dir,file))],axis=0)
            base_file.dropna(inplace=True)
            base_file.drop(["Unnamed: 0"],axis=1,inplace=True)
            base_file.drop_duplicates(inplace=True)
            if not prepare_data(None,train_size,no_of_samples,base_file,name = "full_corpa"):
                return False
        else:
            for file in files:
                if not prepare_data(os.path.join(root_dir,file),train_size=train_size,no_of_samples=no_of_samples):
                    return False
        return True
    except Exception as e:
        logger.error(f"UNABLE TO COMBINE AND SAVE DATASETS:{e}")
        return False