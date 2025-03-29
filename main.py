import re
from simple_tokenizer import SimpleTokenizerV1

def load_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    return raw_text

def tokenize(raw_text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(len(preprocessed))
    print(preprocessed[:30])
    return preprocessed

def create_vocabulary(preprocessed):
    all_words = sorted(set(preprocessed))
    vocab = {token: integer for integer, token in enumerate(all_words)}
    return vocab

def main():
    raw_text = load_file("the-verdict.txt")
    preprocessed = tokenize(raw_text)
    vocab = create_vocabulary(preprocessed)

    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know,"
           Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

if __name__ == "__main__":
    main()
