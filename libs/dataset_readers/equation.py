"""

"""
from typing import Tuple, List, Dict
import re
from collections import Counter

from libs.dataset_readers.equation_utils import check_and_tokenize


def preprocess_equation(equation: str, reserved_numbers: List[str], occurrence_limit: bool = True
                        ) -> List[str]:
    """Tokenizes the equation, and replaces the reserved numbers with pseudo tokens.

    This function tokenizes the input equation while keeping intact the reserved numbers (the numbers from the problem).
    Then it replaces these numbers with pseudo tokens <N0>, <N1>, ....

    Note that in some settings, we only want to replace those numbers of 1 occurrence in the problem text
    because we cannot determine whether the "2" in the equation comes from the first or the second "2" in the problem.
    We would like to use other copy mechanisms to determine these numbers later. In this case, 
    use `occurrence_limit` = True, and the function will ignore the numbers with multiple occurrence in the problem.

    Also note that we need to do the tokenization for fractions and other types of numbers separately 
    because longer frations can be misconsidered as combinations of shorter numbers.
    So it's safer to replace fractions first.

        Usage example
        ------------------------------------
        >>> expression = "(3 + 5 * 2) / (2/3)"
        >>> reserved_numbers = ["3", "(2/3)", "2", "(2/3)"]
        >>> preprocess_equation(expression, reserved_numbers, occurrence_limit=True)
        ['(', '<N0>', '+', '5', '*', '<N2>', ')', '/', '(2/3)']
        >>> preprocess_equation(expression, reserved_numbers, occurrence_limit=False)
        ['(', '<N0>', '+', '5', '*', '<N2>', ')', '/', '<N1>']


    """

    # Replace fractions with pseudo tokens N0, N1 , ...
    # We have to deal with longer fractions first; otherwise they get confused with shorter ones.
    fractions = []
    for num in reserved_numbers:
        if re.search(r"\d*\(\d+/\d+\)\d*", num):
            fractions.append(num)
    fractions = sorted(fractions, key=lambda x: len(x), reverse=True)

    number_to_pseudo = {}
    for fraction in fractions:
        if fraction in equation:
            pseudo_token = f"<N{reserved_numbers.index(fraction)}>"
#             pseudo_token = f"N{reserved_numbers.index(fraction)}"
            equation = equation.replace(fraction, pseudo_token)
            number_to_pseudo[fraction] = pseudo_token

    # Tokenize the equation/expression
    output_tokens = check_and_tokenize(equation)

    # Replace ordinary numbers (i.e., 3 -> <N0>)
    for idx, token in enumerate(output_tokens):
        if token in reserved_numbers:
            number = token
            pseudo_token = f"<N{reserved_numbers.index(number)}>"
            # pseudo_token = f"N{reserved_numbers.index(number)}"
            output_tokens[idx] = pseudo_token
            number_to_pseudo[number] = pseudo_token

    # Map some pseudo tokens back to their numbers if they occur more than one time in the problem
    if occurrence_limit is True:
        counter = Counter(reserved_numbers)
        back_mapping = {number_to_pseudo[num]: num
                        for num in number_to_pseudo
                        if counter[num] >= 2}

        for idx, token in enumerate(output_tokens):
            if token in back_mapping:
                output_tokens[idx] = back_mapping[token]

    return output_tokens
