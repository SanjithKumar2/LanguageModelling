from collections import defaultdict
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