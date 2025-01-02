from typing import Generator
import os
from byte_pair_encoding import bpe_generator


def gutenberg_paragraphs() -> Generator[str, None, None]:
    for root, _, files in os.walk("gutenberg"):
        for file in files:
            print(file)
            file = os.path.join("gutenberg", file)
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                paragraphs = text.split("\n\n")
                target_length = min(max(len(paragraph) for paragraph in paragraphs), 600)
                print(target_length)
                story = ""
                for paragraph in paragraphs[16:]:
                    if "Gutenberg" in paragraph or "[Illustration]" in paragraph:
                        continue
                    if "Footnotes:" in paragraph:
                        if len(story) > target_length//2:
                            yield story
                        break
                    story += paragraph
                    if len(story) >= target_length:
                        yield story
                        story = ""


def gutenberg_tokens():
    tokens = bpe_generator(list(gutenberg_paragraphs()), 2048)
    with open("datasets/gutenberg_tokens.txt", "w") as f:
        f.write(str(tokens))

#gutenberg_tokens()