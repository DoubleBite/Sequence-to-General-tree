import json
import logging
from typing import Any, Dict, List, Tuple, Optional, Iterable, Union

import re

# Fractions such as 1/5, 2(3/4)
FRACTION_PATTERN = r"\d*\(\d+/\d+\)\d*"

# Decimals such as 2.5 or 2.5%
DECIMAL_PATTERN = r"\d+\.\d+%?"

# Integers such as 25 or 25%
INTEGER_PATTERN = r"\d+%?"

# All patterns
NUMBER_PATTERN = (FRACTION_PATTERN + r"|" +
                  DECIMAL_PATTERN + r"|" + INTEGER_PATTERN)


def preprocess_text(text: str, replace_type: str = "NUM"
                    ) -> Tuple[List[str], List[str], List[int]]:
    """

    Tokenizes the text, collects its numbers, and replaces these numbers with pseudo tokens(<N0>, <N1>, ...).

        Usage example
        ------------------------------------
        # Replace with <NUM>    
        >>> text = "有 男生 25 人 ， 女生 23 人 ． 体育课 上 ， 周 老师 把 他们 每 8 人 站 一队 ， 一共 可以 站 多少 队 ?"
        >>> tokens, numbers, positions = preprocess_text(text, replace_type="NUM") 
        >>> tokens # doctest: +ELLIPSIS
        ['有', '男生', '<NUM>', '人', '，', '女生', '<NUM>', '人', ..., '每', '<NUM>', '人', '站', '一队', '，', ...]
        >>> numbers
        ['25', '23', '8']
        >>> positions
        [2, 6, 17]

        # Replace with Nx    
        >>> tokens, numbers, positions = preprocess_text(text, replace_type="Nx") 
        >>> tokens # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        ['有', '男生', '<N0>', '人', '，', '女生', '<N1>', '人', ..., '每', '<N2>', '人', '站', '一队', '，', ...]
        >>> numbers
        ['25', '23', '8']
        >>> positions
        [2, 6, 17]

        # No replacement 
        >>> tokens, numbers, positions = preprocess_text(text, replace_type=None)
        >>> tokens # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        ['有', '男生', '25', '人', '，', '女生', '23', '人', ..., '每', '8', '人', '站', '一队', '，', ...]
        >>> numbers
        []
        >>> positions
        []

    Args: 
        text: `string`, required
            The problem text
        replace_type: `string`, optional (default="[NUM]")
            The type of pseudo token to replace, only three choices available: `NUM`, `Nx`, `None`.
            If the value is "NUM", all the numbers will be replaced with "<NUM>".
            If the value is "Nx", the numbers will be replaced with "<N0>", "<N1>", "<N2>", and so on.
            If the value is None, we don't not do any replacement.

    Returns:
        output_tokens: `List[str]`
            The output tokens consist of tokenized words.
        numbers: `List[str]`
            The numbers in the text.
        positions: `List[int]`
            The position of the numbers in the text.

    """
    # Base checks
    assert replace_type in ["NUM", "Nx", None]

    input_tokens = text.strip().split(" ")
    output_tokens = []
    numbers = []
    positions = []

    # No replacement
    if replace_type is None:
        return input_tokens, [], []

    # Other cases
    count = 0
    for token in input_tokens:
        match = re.search(NUMBER_PATTERN, token)
        if match and match.start() == 0:  # start==0 meant to filter out "MP3" or the like
            if replace_type == "NUM":
                pseudo_token = "<NUM>"
                # pseudo_token = "NUM"
            elif replace_type == "Nx":
                pseudo_token = f"<N{count}>"
                count += 1
            numbers.append(token[match.start(): match.end()])
            positions.append(len(output_tokens))
            output_tokens.append(pseudo_token)
            if match.end() < len(token):
                output_tokens.append(token[match.end():])
        else:
            output_tokens.append(token)
    return output_tokens, numbers, positions
