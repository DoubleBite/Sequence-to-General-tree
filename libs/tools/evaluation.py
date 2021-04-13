import re
from libs.tools.formula import evaluate_custom_function, FUNC_NAMES, NUM_OF_ARGS


def replace_pseudo_tokens(prefix_tokens, numbers):
    """

        Example usage
        ---------------------------------------
        >>> replace_pseudo_tokens(["*", "<N0>", "<N1>"], ['5', '2'])
        ['*', '5', '2']
    """
    result_tokens = []
    for token in prefix_tokens:
        if re.search(r"<N\d>", token):
            if int(token[2:-1]) >= len(numbers):
                token = "0"
            else:
                token = numbers[int(token[2:-1])]
        result_tokens.append(token)
    return result_tokens


def evaluate_number(number_string):
    """


    """
    if re.search(r"\d+\(", number_string):
        pos = re.search(r"\d+\(", number_string)
        return eval(number_string[pos.start(): pos.end() - 1] + "+" + number_string[pos.end() - 1:])
    elif number_string[-1] == "%":
        return float(number_string[:-1]) / 100
    else:
        try:
            return eval(number_string)
        except:
            return 0  # This is due to data parsing errors, should be fixed


def evaluate_prefix(prefix_tokens):
    """

        Example usage
        >>> evaluate_prefix(["+", "5.0", "2.2"])
        7.2
        >>> evaluate_prefix(["triangle_area", "5", "2"])
        5.0
        >>> evaluate_prefix(["square_area", "5", "2"])
        'NaN'

    """
    operand_stack = list()
    operators = ["+", "-", "^", "*", "/"]
    reversed_prefix_tokens = reversed(prefix_tokens)

    for token in reversed_prefix_tokens:

        if token not in operators and token not in FUNC_NAMES:
            # Numbers like: 5(2/3), 100%, 10
            operand_stack.append(evaluate_number(token))
        elif token in FUNC_NAMES and len(operand_stack) >= NUM_OF_ARGS[token]:
            args = [operand_stack.pop() for _ in range(NUM_OF_ARGS[token])]
            func_name = token
            value = evaluate_custom_function(func_name, args)
            operand_stack.append(value)
        elif token == "+" and len(operand_stack) > 1:
            arg1 = operand_stack.pop()
            arg2 = operand_stack.pop()
            operand_stack.append(arg1 + arg2)
        elif token == "-" and len(operand_stack) > 1:
            arg1 = operand_stack.pop()
            arg2 = operand_stack.pop()
            operand_stack.append(arg1 - arg2)
        elif token == "*" and len(operand_stack) > 1:
            arg1 = operand_stack.pop()
            arg2 = operand_stack.pop()
            operand_stack.append(arg1 * arg2)
        elif token == "/" and len(operand_stack) > 1:
            arg1 = operand_stack.pop()
            arg2 = operand_stack.pop()
            if arg2 == 0:
                return "NaN"
            operand_stack.append(arg1 / arg2)
        elif token == "^" and len(operand_stack) > 1:
            arg1 = operand_stack.pop()
            arg2 = operand_stack.pop()
            operand_stack.append(arg1 ** arg2)
        else:
            return "NaN"

    if len(operand_stack) == 1:
        return operand_stack.pop()
    else:
        return "NaN"
