"""
References:
    1. Infix to postfix:
        https://runestone.academy/runestone/books/published/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html
    2. Infix to prefix:
        https://www.geeksforgeeks.org/convert-infix-prefix-notation/

"""

import re


def check_and_tokenize(expression):
    """Tokenizes the expression string into tokens.

    If it is already a list, then just ruturn it. If it is anything else, then we raise an exception.

        Example usage
        ---------------------------------
        >>> check_and_tokenize("2+3^(3-1)/4")
        ['2', '+', '3', '^', '(', '3', '-', '1', ')', '/', '4']
        >>> check_and_tokenize(['2', '+', '3', '^', '3', '/', '4'])
        ['2', '+', '3', '^', '3', '/', '4']
        >>> check_and_tokenize("420.08*2.3-420.08")
        ['420.08', '*', '2.3', '-', '420.08']
        >>> check_and_tokenize("420.08*<N0>")
        ['420.08', '*', '<N0>']
        >>> check_and_tokenize("rectangle_area(8, 18)")
        ['rectangle_area', '(', '8', ',', '18', ')']
        >>> check_and_tokenize(dict())
        Traceback (most recent call last):
            ...
        ValueError: The input expression cannot be processed.

    """
    if isinstance(expression, list):
        return expression
    elif isinstance(expression, str):
        expression = expression.replace(" ", "")
        tokens = re.split(r"([\[\]()+\-*/^,])", expression)
        tokens = [x for x in tokens if x]  # Get rid of empty tokens
        return tokens
    else:
        raise ValueError("The input expression cannot be processed.")


def swap_parentheses(token):
    if token == "(":
        return ")"
    elif token == "[":
        return "]"
    elif token == ")":
        return "("
    elif token == "]":
        return "["
    else:
        return token


def convert_expression_notation(expression_tokens, notation_type):
    assert notation_type in ["prefix", "infix", "postfix"]
    if notation_type == "infix":
        return expression_tokens
    elif notation_type == "prefix":
        return infix_to_prefix(expression_tokens)
    elif notation_type == "postfix":
        return infix_to_postfix(expression_tokens)


def _infix_to_postfix(expression_tokens):
    """The low-level conversion function for `infix_to_postfix()` and `infix_to_prefix()`

    See [1] for details of the algorithm.

        Example usage
        --------------------------------
        >>> _infix_to_postfix(["A", "+", "B"])
        ['A', 'B', '+']
        >>> _infix_to_postfix(["[", "(", "A", "+", "B", ")", "^", "(", "A", "-", "B", ")", "]", "/", "2"])
        ['A', 'B', '+', 'A', 'B', '-', '^', '2', '/']

    """
    op_stack = list()
    postfix_tokens = list()

    op_priority = {
        "(": -1,
        "[": -1,
        ",": 0,  # For custom functions, we need to pop all the ops when encoutering a comma
        "+": 1,
        "-": 1,
        "*": 2,
        "/": 2,
        "^": 3,
    }

    for token in expression_tokens:
        if token in ["(", "["]:
            op_stack.append(token)
        elif token in op_priority:  # +-*/^([
            while (len(op_stack) > 0) and (op_priority[token] < op_priority[op_stack[-1]]):
                postfix_tokens.append(op_stack.pop())
            op_stack.append(token)
        elif token == ")":
            while len(op_stack) > 0:
                last_op = op_stack.pop()
                if last_op == "(":
                    break
                else:
                    postfix_tokens.append(last_op)
        elif token == "]":
            while len(op_stack) > 0:
                last_op = op_stack.pop()
                if last_op == "[":
                    break
                else:
                    postfix_tokens.append(last_op)
        else:  # Numbers or variables
            postfix_tokens.append(token)

    # Place the remaining operators into the result
    while len(op_stack) > 0:
        postfix_tokens.append(op_stack.pop())

    return postfix_tokens


def infix_to_postfix(expression):
    """Convert an infix expression to postfix.

    We also need to deal with custom formulas in the expressions like "triangle_area(5, 2) + 32". 
    Here we just make some simple transformation "triangle_area(5, 2) --> (5, 2)triangle_area",
    so that it can be processed in postfix conversion seamlessly, i.e., (5, 2)triangle_area to [5, 2, triangle_area].

        Example usage
        --------------------------------
        >>> infix_to_postfix("A+B")
        ['A', 'B', '+']
        >>> infix_to_postfix("[(A+B)^(A-B)]/2")
        ['A', 'B', '+', 'A', 'B', '-', '^', '2', '/']
        >>> infix_to_postfix(" [rectangle_area(A, B) ^ (A-B)]/2")
        ['A', 'B', 'rectangle_area', 'A', 'B', '-', '^', '2', '/']
    """
    # Preprocessing: triangle_area(5, 2) --> (5, 2)triangle_area
    expression = re.sub(r"([a-z_]+)(\([\w\s,.]+\))", r"\2\1", expression)

    expression_tokens = check_and_tokenize(expression)

    postfix_tokens = _infix_to_postfix(expression_tokens)
    postfix_tokens = [
        x for x in postfix_tokens if x != ","]  # Remove commas

    return postfix_tokens


def infix_to_prefix(expression):
    """Convert an infix expression to prefix.

    See [2] for details of the algorithm.

    We also need to deal with custom formulas in the expressions like "triangle_area(5, 2) + 32".
    In this case, we just tokenize the formula and remove the commas because it is already in prefix order.  

        Example usage
        --------------------------------
        >>> infix_to_prefix("A+B")
        ['+', 'A', 'B']
        >>> infix_to_prefix("[(A+B)^(A-B)]/2")
        ['/', '^', '+', 'A', 'B', '-', 'A', 'B', '2']
        >>> infix_to_prefix("[cuboid_volume(A, B, C)^(A-B)]/2")
        ['/', '^', 'cuboid_volume', 'A', 'B', 'C', '-', 'A', 'B', '2']
        >>> infix_to_prefix("rectangle_perimeter(8+5*2, 8^3-2)")
        ['rectangle_perimeter', '+', '8', '*', '5', '2', '-', '^', '8', '3', '2']
    """

    expression_tokens = check_and_tokenize(expression)

    # 1. Reverse the tokens and swap the right and left parentheses
    reversed_expression_tokens = list(reversed(expression_tokens))
    reversed_expression_tokens = [swap_parentheses(x)
                                  for x in reversed_expression_tokens]
    # 2. Do postfix conversion
    reversed_postfix_tokens = _infix_to_postfix(reversed_expression_tokens)
    reversed_postfix_tokens = [
        x for x in reversed_postfix_tokens if x != ","]  # Remove commas

    # 3. Reverse it back
    infix_tokens = list(reversed(reversed_postfix_tokens))

    return infix_tokens
