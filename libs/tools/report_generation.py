"""
# 可能還需要算 statistics
# 可能需要把 others 的東西列出來
用 counter 來搞吧

處理 syntax error
unbound local error


"""
import os
from typing import Tuple, List, Dict

import re
from collections import Counter
from graphviz import Source
from lib.tools.visualization import create_dot_tree


def generate_markdown_report(problems: List[Dict],
                             report_path: str,
                             report_statistics: bool = False,
                             render_tree: bool = False,
                             img_directory: str = "tmp"):
    """
    """
    statistics = {}
    statistics["num_problem"] = len(problems)
    statistics["with_knowledge"] = 0
    statistics["shape_distribution"] = Counter()
    statistics["formula_distribution"] = Counter()

    # Calculate the statistics if necessary
    if report_statistics is True:
        for problem in problems:
            if len(problem["knowledge"]) > 0:
                statistics["with_knowledge"] += 1
                formulas = problem["knowledge"]
                shapes = list(
                    map(lambda x: x.strip("()").split(",")[0], formulas))
                statistics["shape_distribution"].update(shapes)
                statistics["formula_distribution"].update(formulas)

    # Write out the report
    with open(report_path, 'w') as f:

        # Write out the statistics
        if report_statistics is True:

            print(f"+ Total problems: {statistics['num_problem']}", file=f)
            print(f"+ With knowledge: {statistics['with_knowledge']}", file=f)

            print("+ Shape distribution:", file=f)
            for shape, count in statistics["shape_distribution"].most_common():
                print(
                    f"\t+ {shape}: {count} ({count/statistics['with_knowledge']*100:.2f}%)",
                    file=f)

            print("+ Formula distribution:", file=f)
            for formula, count in statistics["formula_distribution"].most_common():
                print(
                    f"\t+ {formula}: {count} ({count/statistics['with_knowledge']*100:.2f}%)",
                    file=f)

            print("<br>\n", file=f)
            print(
                "============================================================", file=f)
            print(
                "============================================================\n", file=f)
            print("<br>", file=f)

        # Write out the problems
        for problem in problems:

            # Dump the poroblem
            try:
                markdown_prob = markdown_single_problem(
                    problem, render_tree=render_tree)
                print(markdown_prob, file=f)
            except (SyntaxError, UnboundLocalError):
                pass

            # Add borderlines
            print("<br>", file=f)
            print("---", file=f)
            print("<br>", file=f)


#################################################################
# Template and the Function for single problem
#################################################################
PROBLEM_TEMPLATE = """
{id}.  {problem_text}  
+ Equation: {equation}
+ Answer :  {answer}
+ Knowledge:
    + {knowledge}
+ History:
    + {history}

![No Image]({image})
"""


def markdown_single_problem(problem: Dict,
                            render_tree: bool = False,
                            img_directory: str = "tmp"
                            ) -> str:
    """Given a problem, wraps it in markdown format.

    This function wraps a problem of json/dict format into markdown format.
    It takes a keyword argument, `render_tree` to decide whether to reander the expression tree or not.

    Args
        problem: `dict`
            A problem in the math23k or math23k-KE format
        render_tree: `bool`, optional (default="False)
            Whether to render the expression tree. If not, the markdown image will show the alt text.
        img_directory: `str`, optional (default="tmp)
            The directory to save the tree images. It is used when "render_tree" is True 

    """
    # Basic information
    p_id = problem["id"]
    text = problem["original_text"]
    equation = problem["equation"]
    answer = problem["ans"]

    # Knowledge, history, and image
    knowledge = None
    history = None
    image = None

    if "knowledge" in problem:
        knowledge_formulas = problem["knowledge"]
        knowledge = "\n \t+ ".join(knowledge_formulas)

    if "history" in problem:
        history_equations = problem["history"]
        history = "\n \t+ ".join(history_equations)

    if render_tree == True:
        expression = equation[2:]
        tree = create_dot_tree(expression)
        tree = Source(tree)
        image = tree.render(filename=f"{p_id}", format="svg",
                            cleanup=True, directory=img_directory)

    return PROBLEM_TEMPLATE.format(
        id=p_id,
        problem_text=text,
        equation=escape_asterisk(equation),
        answer=answer,
        knowledge=knowledge,
        history=escape_asterisk(history),
        image=image
    )


#################################################################
# Utility functions
#################################################################

def escape_asterisk(string: str):
    if string:
        return re.sub(r"\*", r"\*", string)
    else:
        return None
