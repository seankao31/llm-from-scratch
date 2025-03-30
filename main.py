import tiktoken

def load_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    return raw_text

def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)

    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(ids)
    print(tokenizer.decode(ids))

if __name__ == "__main__":
    main()
