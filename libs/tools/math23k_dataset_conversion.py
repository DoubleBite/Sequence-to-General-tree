"""This module provides the following function(s) to reannotate the geometry problems.

    1. reannotate_geometry_problem()

Specifically, the `reannotate_geometry_problem()` function checks 
whether the equation contains the pattern of certain geoemtry formula 
and the text contains keywords for that formula. If the problem does, 
it will `reannotate` the problem. See the example below.

    Usage example
    -----------------------------------
    >>> problem = {"original_text": "一个正方形的边长是10厘米，如果边长减少4厘米，则它的面积减少多少平方厘米．",
    ...            "equation": "x=10*10-(10-4)*(10-4)"}
    >>> reannotate_geometry_problem(problem) # doctest: +NORMALIZE_WHITESPACE
    {'original_text': '一个正方形的边长是10厘米，如果边长减少4厘米，则它的面积减少多少平方厘米．',
        'equation': 'x=square_area(10)-square_area((10-4))',
        'knowledge': ['(square, side * side = area)'],
        'history': ['x=10*10-(10-4)*(10-4)']}


References:
    [1]. Stackoverflow - How to capture the repeating operands of a multiplication using python regex:
        https://stackoverflow.com/questions/64908314/how-to-capture-the-repeating-operands-of-a-multiplication-using-python-regex

"""
from typing import Tuple, List, Dict
from tqdm import tqdm

import re
from copy import deepcopy


def reannotate_geometry_problem(problem: dict):
    """Reannotate the geometry problem if it is considered to use certain geometry formulas

    Specifically, this function reannotates the equation by substituting meaningful formula representations
    into the original equation, e.g., "4*5/2" --> "triangle_area(4, 5)". This function also puts 
    the original equation into `history` entry and appends the formula being used into `knowledge` entry.

    It checks each shape one by one. Note that the order matters because in some cases the patterns for 
    different shapes are quite similar, e.g., "a*b/2" for triangle area and "a*b" for rectangle area.
    So we have to check from shapes that have the most determined pattern. 
    Also, we have to try each shape because a problem may contain more than two formulas.

    """
    problem = normalize_problem(problem)

    # Check each shape one by one
    # See further details for each shape from their definitions.
    problem = reannotate_square_problem(problem)
    problem = reannotate_circle_problem(problem)
    problem = reannotate_triangle_problem(problem)
    problem = reannotate_rectangle_problem(problem)

    # Here we address the complex problems that are not covered by our rules
    problem = hard_annotate(problem)

    return problem


def normalize_problem(problem: dict):
    """Helper function for copying the input problem and adding some auxiliary entries.
    """
    problem = deepcopy(problem)
    if "knowledge" not in problem:
        problem["knowledge"] = []
    if "history" not in problem:
        problem["history"] = []
    return problem


######################################################################
# FUNCTIONS for each shape
######################################################################


def reannotate_square_problem(problem: dict):
    """Re-annotate the equation and important infomation if the problem uses formulas related to `square/cubic`.

    This function checks whether the problem uses the formulas for the important quantities of a square and cubic,
    including:
        (1) cubic volume, (2) square area, (3) square perimeter

    Each quantity corresponds to a set of keywords, a regular expression pattern, and a formula.
    If the text contains the keywords and the equation contains the regular expression pattern,
    we re-annotate the equation by substituting the formula into the corresponding part of the equation .
    We also add the original equation into `history` entry and the formula into `knowledge` entry.
    For convenience, the returned dict is the same as the input dict with all the changes in place.

    Note that we have to check `cubic volume` before `square area` because the pattern for cubic volume (A*A*A)
    can be misconsidered as the pattern for square area (A*A).


        Examples
        ---------------------
        [Square area]
        >>> problem = {"original_text": "一个正方形的边长是10厘米，如果边长减少4厘米，则它的面积减少多少平方厘米．",
        ...            "equation": "x=10*10-(10-4)*(10-4)"}
        >>> problem = normalize_problem(problem)
        >>> reannotate_square_problem(problem) # doctest: +NORMALIZE_WHITESPACE
        {'original_text': '一个正方形的边长是10厘米，如果边长减少4厘米，则它的面积减少多少平方厘米．',
         'equation': 'x=square_area(10)-square_area((10-4))',
         'knowledge': ['(square, side * side = area)'],
         'history': ['x=10*10-(10-4)*(10-4)']}

        [Cubic volume]
        >>> problem = {"original_text": "把一个棱长6分米的正方体钢坯，锻造成一个底面积是9平方分米的长方体钢块，能锻造多厚？",
        ...            "equation": 'x=6*6*6/9'}
        >>> problem = normalize_problem(problem)
        >>> reannotate_square_problem(problem) # doctest: +NORMALIZE_WHITESPACE
        {'original_text': '把一个棱长6分米的正方体钢坯，锻造成一个底面积是9平方分米的长方体钢块，能锻造多厚？',
         'equation': 'x=cubic_volume(6)/9',
         'knowledge': ['(cubic, side * side * side = area)'],
         'history': ['x=6*6*6/9']}

        [Square perimeter]
        >>> problem = {"original_text": "一个正方形的边长是4分米，它的周长=．",
        ...            "equation": 'x=4*4'}
        >>> problem = normalize_problem(problem)
        >>> reannotate_square_problem(problem) # doctest: +NORMALIZE_WHITESPACE
        {'original_text': '一个正方形的边长是4分米，它的周长=．',
         'equation': 'x=square_peri(4)',
         'knowledge': ['(square, side * 4 = perimeter)'],
         'history': ['x=4*4']}
    """

    CRITERIA = {
        "volume": {
            "keywords": [("正方体")],
            # A*A*A or (A+B)*(A+B)*(A+B)
            "pattern": r"(?<!\d)([\d.]+)\*\1\*\1(?!\d)|(\([\d+-.]+\))\*\2\*\2"
        },
        "area": {
            "keywords": [("正方", "面积"), ("正方", "每平方")],
            # A*A or (A+B)*(A+B), see [1] for reference.
            "pattern": r"(?<!\d)([\d.]+)\*\1(?!\d)|(\([\d+-/*.]+\))\*\2"
        },
        "perimeter": {
            "keywords": [("正方", "周长"), ("正方", "广场")],
            # 4*A or A*4
            "pattern": r"(?<!\d)([\d.]+)\*4(?!\d)|(?<!\d)4\*([\d.]+)(?!\d)"
        },
    }

    FORMULAS = {
        "volume": "(cubic, side * side * side = area)",
        "area": "(square, side * side = area)",
        "perimeter": "(square, side * 4 = perimeter)",
    }

    # If the problem text contains keywords and the equation contains patterns of certain formula
    for quan_type in ["volume", "area", "perimeter"]:

        keyword_sets = CRITERIA[quan_type]["keywords"]
        pattern = CRITERIA[quan_type]["pattern"]

        if contain_keywords(problem["original_text"], keyword_sets) \
                and re.findall(pattern, problem["equation"]):

            # Add the formula used in the equation to `knowledge`
            # Add the original equation to `history`
            problem["knowledge"].append(FORMULAS[quan_type])
            problem["history"].append(problem["equation"])

            # Make equation substitution
            if quan_type == "volume":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"cubic_volume({m.group(1) or m.group(2)})", problem["equation"])
            elif quan_type == "area":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"square_area({m.group(1) or m.group(2)})", problem["equation"])
            elif quan_type == "perimeter":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"square_peri({m.group(1) or m.group(2)})", problem["equation"])

    return problem


def reannotate_circle_problem(problem: dict):
    """Re-annotate the equation and important infomation if the problem uses formulas related to `circle`.

    This function checks whether the problem uses the formulas for the important quantities of a circle,
    including:
        (1) circle area, (2) circumference

    Each quantity corresponds to a set of keywords, a regular expression pattern, and a formula.
    If the text contains the keywords and the equation contains the regular expression pattern,
    we re-annotate the equation by substituting the formula into the corresponding part of the equation.
    We also add the original equation into `history` entry and the formula into `knowledge` entry.
    For convenience, the returned dict is the same as the input dict with all the changes in place.

    Note here we use different keywords and patterns for `area` and `the difference of area`,
    and `circumference related to radius` and `circumference related to diameter`.

        Examples
        ---------------------
        # Area
        >>> problem = {"original_text": "一个圆的半径是3厘米，如果把它的半径延长2厘米，那么面积增加多少．",
        ...            "equation": "x=3.14*[(3+2)^2-3^2]"}
        >>> problem = normalize_problem(problem)
        >>> reannotate_circle_problem(problem) # doctest: +NORMALIZE_WHITESPACE
        {'original_text': '一个圆的半径是3厘米，如果把它的半径延长2厘米，那么面积增加多少．',
         'equation': 'x=circle_area((3+2)) - circle_area(3)',
         'knowledge': ['(circle, 3.14 * radius^2 = area)'],
         'history': ['x=3.14*[(3+2)^2-3^2]']}

        # Calculate circumference from radius
        >>> problem = {"original_text": "画圆时，圆规两脚叉开的距离是2cm，画出的圆的周长=多少cm．",
        ...            "equation": "x=3.14*2*2"}
        >>> problem = normalize_problem(problem)
        >>> reannotate_circle_problem(problem) # doctest: +NORMALIZE_WHITESPACE
        {'original_text': '画圆时，圆规两脚叉开的距离是2cm，画出的圆的周长=多少cm．',
         'equation': 'x=circumference_radius(2)',
         'knowledge': ['(circle, 2 * 3.14 * radius = circumference)'],
         'history': ['x=3.14*2*2']}

        # Calculate circumference from diameter
        >>> problem = {"original_text": "一个圆柱的底面直径是2分米，侧面展开图是正方形，这个圆柱的侧面积=？",
        ...            "equation": "x=(3.14*2)*(3.14*2)"}
        >>> problem = normalize_problem(problem)
        >>> problem = reannotate_square_problem(problem)
        >>> reannotate_circle_problem(problem) # doctest: +NORMALIZE_WHITESPACE
        {'original_text': '一个圆柱的底面直径是2分米，侧面展开图是正方形，这个圆柱的侧面积=？',
         'equation': 'x=square_area((circumference_diameter(2)))',
         'knowledge': ['(square, side * side = area)', '(circle, 3.14 * diameter = circumference)'],
         'history': ['x=(3.14*2)*(3.14*2)', 'x=square_area((3.14*2))']}

    """

    CRITERIA = {
        "area": {
            "keywords": [("圆", "面积"), ("圆", "半径")],
            # 3.14*r^2 or 3.14*(A+B)^2
            "pattern": r"(?<!\d)3\.14\*([\d.]+)\^2(?!\d)|(?<!\d)3\.14\*\(([\d+-/*.]+)\)\^2(?!\d)",
        },
        "area_dif": {
            "keywords": [("圆", "面积")],
            # 3.14*[(a+b)^2-(a-b)^2] or 3.14*(r1^2-r2^2)
            "pattern": (r"(?<!\d)3\.14\*\[([(\d+-/*.)]+)\^2[+-]([(\d+-/*.)]+)\^2\](?!\d)"
                        r"|"
                        r"(?<!\d)3\.14\*\(([\d+-/*.]+)\^2[+-]([\d+-/*.]+)\^2\)(?!\d)")
        },
        "circumference_r": {
            "keywords": [("圆", "半径"), ("圆", "周长")],
            # (2*3.14*r) or (3.14*r*2)
            "pattern": r"(?<!\d)2\*3\.14\*([\d.]+)(?!\d)|(?<!\d)3\.14\*([\d.]+)\*2(?!\d)"
        },
        "circumference_d": {
            "keywords": [("圆", "直径")],
            # (3.14*d)
            "pattern": r"(?<!\d)3\.14\*([\d.]+)(?!\d)"
        },
    }

    FORMULAS = {
        "area": "(circle, 3.14 * radius^2 = area)",
        "area_dif": "(circle, 3.14 * radius^2 = area)",
        "circumference_r": "(circle, 2 * 3.14 * radius = circumference)",
        "circumference_d": "(circle, 3.14 * diameter = circumference)",
    }

    # If the problem text contains keywords and the equation contains patterns of certain formula
    for quan_type in ["area", "area_dif", "circumference_r", "circumference_d"]:

        keyword_sets = CRITERIA[quan_type]["keywords"]
        pattern = CRITERIA[quan_type]["pattern"]

        if contain_keywords(problem["original_text"], keyword_sets) \
                and re.findall(pattern, problem["equation"]):

            # Add the formula used in the equation to `knowledge`
            # Add the original equation to `history`
            problem["knowledge"].append(FORMULAS[quan_type])
            problem["history"].append(problem["equation"])

            # Make equation substitution
            if quan_type == "area":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"circle_area({m.group(1) or m.group(2)})", problem["equation"])
            elif quan_type == "area_dif":
                problem["equation"] = re.sub(
                    pattern,
                    lambda m: f"circle_area({m.group(1) or m.group(3)}) - circle_area({m.group(2) or m.group(4)})",
                    problem["equation"])
            elif quan_type == "circumference_r":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"circumference_radius({m.group(1) or m.group(2)})", problem["equation"])
            elif quan_type == "circumference_d":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"circumference_diameter({m.group(1)})", problem["equation"])

    return problem


def reannotate_triangle_problem(problem: dict):
    """Re-annotate the equation and important infomation if the problem uses formulas related to `triangle`.

    This function checks whether the problem uses the formulas for the important quantities of a triangle,
    including:
        (1) triangle area

    Each quantity corresponds to a set of keywords, a regular expression pattern, and a formula.
    If the text contains the keywords and the equation contains the regular expression pattern,
    we re-annotate the equation by substituting the formula into the corresponding part of the equation.
    We also add the original equation into `history` entry and the formula into `knowledge` entry.
    For convenience, the returned dict is the same as the input dict with all the changes in place.

        Examples
        ---------------------
        [Triangle area]
        >>> problem = {"original_text": "一个三角形底是10厘米，是高的2倍，这个三角形的面积=多少平方厘米．",
        ...            "equation": "x=10*(10/2)/2"}
        >>> problem = normalize_problem(problem)
        >>> reannotate_triangle_problem(problem) # doctest: +NORMALIZE_WHITESPACE
        {'original_text': '一个三角形底是10厘米，是高的2倍，这个三角形的面积=多少平方厘米．',
         'equation': 'x=triangle_area(10, (10/2))',
         'knowledge': ['(triangle, base * height / 2 = area)'],
         'history': ['x=10*(10/2)/2']}
    """

    CRITERIA = {
        "area": {
            "keywords": [("三角", "面积")],
            # a*b/2 or (a+b)*(c+d)/2
            "pattern": r"(?<!\d)([\d.]+|\([\d+-/*.]+\))\*([\d.]+|\([\d+-/*.]+\))/2(?!\d)"
        }
    }

    FORMULAS = {
        "area": "(triangle, base * height / 2 = area)",
    }

    # If the problem text contains keywords and the equation contains patterns of certain formula
    for quan_type in ["area"]:

        keyword_sets = CRITERIA[quan_type]["keywords"]
        pattern = CRITERIA[quan_type]["pattern"]

        if contain_keywords(problem["original_text"], keyword_sets) \
                and re.findall(pattern, problem["equation"]):

            # Add the formula used in the equation to `knowledge`
            # Add the original equation to `history`
            problem["knowledge"].append(FORMULAS[quan_type])
            problem["history"].append(problem["equation"])

            # Make equation substitution
            if quan_type == "area":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"triangle_area({m.group(1)}, {m.group(2)})", problem["equation"])

    return problem


def reannotate_rectangle_problem(problem: dict):
    """Re-annotate the equation and important infomation if the problem uses formulas related to `rectangle/cuboid`.

    This function checks whether the problem uses the formulas for the important quantities of a rectangle or a cuboid,
    including:
        (1) cuboid volume, (2) cuboid surface area, (3) rectangle perimeter, (4) rectangle area

    Each quantity corresponds to a set of keywords, a regular expression pattern, and a formula.
    If the text contains the keywords and the equation contains the regular expression pattern,
    we re-annotate the equation by substituting the formula into the corresponding part of the equation.
    We also add the original equation into `history` entry and the formula into `knowledge` entry.
    For convenience, the returned dict is the same as the input dict with all the changes in place.

    Note that we seperate two different scenarios for calculating the area of a rectangle (for keeping RE patterns simple)


        Examples
        ---------------------
        [Cuboid volume + rectangle area]
        >>> problem = {"original_text": ("一个长方体水箱，长为8分米，宽为6分米，高为5分米，装水高4分米，现在把一块长3分米"
        ...                              "，宽2分米的长方体铁块浸没在水箱中，这时水面高4.2分米．求这个铁块的高=多少分米？"),
        ...            "equation": "x=8*6*(4.2-4)/(3*2)"}
        >>> problem = normalize_problem(problem)
        >>> reannotate_rectangle_problem(problem) # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        {'original_text': '一个长方体水箱，长为8分米，宽为6分米，...求这个铁块的高=多少分米？',
         'equation': 'x=cuboid_volume(8, 6, (4.2-4))/rectangle_area(3, 2)',
         'knowledge': ['(cuboid, length * width * height = volume)', '(rectangle, length * width = area)'],
         'history': ['x=8*6*(4.2-4)/(3*2)', 'x=cuboid_volume(8, 6, (4.2-4))/(3*2)']}

        [Cuboid surface area]
        >>> problem = {"original_text": "有一个长4厘米，宽3厘米，高2厘米的长方体，它的表面积=多少平方厘米．",
        ...            "equation": "x=(4*3+4*2+3*2)*2"}
        >>> problem = normalize_problem(problem) 
        >>> reannotate_rectangle_problem(problem) # doctest: +NORMALIZE_WHITESPACE
        {'original_text': '有一个长4厘米，宽3厘米，高2厘米的长方体，它的表面积=多少平方厘米．',
         'equation': 'x=cuboid_surface(4, 3, 2)',
         'knowledge': ['(cuboid, (length * width + length * height + width * height) * 2 = surface_area)'],
         'history': ['x=(4*3+4*2+3*2)*2']}

        [Rectangle perimeter]
        >>> problem = {"original_text": "长方形的长是8cm，宽比长短2cm，这个长方形的周长=．",
        ...            "equation": "x=(8+8-2)*2"}
        >>> problem = normalize_problem(problem) 
        >>> reannotate_rectangle_problem(problem) # doctest: +NORMALIZE_WHITESPACE
        {'original_text': '长方形的长是8cm，宽比长短2cm，这个长方形的周长=．',
         'equation': 'x=rectangle_perimeter(8, 8-2)',
         'knowledge': ['(rectangle, (length + width) * 2 = perimeter)'],
         'history': ['x=(8+8-2)*2']}
    """

    CRITERIA = {
        "volume": {
            "keywords": [("长方体", "长", "宽", "高", "^表面积"), ("长方体", "体积=")],
            # a*b*c or (a+b)*(c+d)*(e+f)
            "pattern": r"(?<!\d)([\d.]+|\([\d+-/*.]+\))\*([\d.]+|\([\d+-/*.]+\))\*([\d.]+|\([\d+-/*.]+\))(?!\d)",
        },
        "surface": {
            "keywords": [("长方体", "表面积")],
            # (a*b+a*c+b*c)*2
            "pattern": r"(?<!\d)\(([\d.]+)\*([\d.]+)\+\1\*([\d.]+)\+\2\*\3\)\*2(?!\d)"
        },
        "perimeter": {
            "keywords": [("长方", "周长")],
            # (a+b)*2
            "pattern": r"(?<!\d)\(([\d+-/*.]+)\+([\d+-/*.]+)\)\*2(?!\d)"
        },
        "area1": {
            "keywords": [("长方", "高="), ("长方", "长="), ("长方", "多长"), ("长方", "宽="),
                         ("长方", "上升多少"), ("长方", "下降多少"), ("长方", "降低多少"),
                         ("长方", "多深"), ("长方", "深多少"), ("长方", "深="), ("长方", "深度="),
                         ("长方", "多厚")],
            # (...)/(a*b)
            "pattern": r"(?<!\d)\(([\d.]+)\*([\d.]+)\)(?!\d)"
        },
        "area2": {
            "keywords": [("长方体", "切面"), ("长方体", "面积="), ("长方体", "多少平方"), ("长方形", "长", "宽")],
            # a*b or (a+b)*(c+d)
            # We should give a+b a higher order priority, otherwise it will turn "(8*2+6*2)*2" into Area(8*2+6*2,2)
            "pattern": (r"(?<!\d)([\d.]+)\*([\d.]+)(?!\d)"
                        r"|"
                        r"(?<!\d)([\d.]+|\([\d+-.]+\))\*([\d.]+|\([\d+-.]+\))(?!\d)")
        }
    }

    FORMULAS = {
        "volume": "(cuboid, length * width * height = volume)",
        "surface": "(cuboid, (length * width + length * height + width * height) * 2 = surface_area)",
        "perimeter": "(rectangle, (length + width) * 2 = perimeter)",
        "area1": "(rectangle, length * width = area)",
        "area2": "(rectangle, length * width = area)",
    }

    # If the problem text contains keywords and the equation contains patterns of certain formula
    for quan_type in ["volume", "surface", "perimeter", "area1", "area2"]:

        keyword_sets = CRITERIA[quan_type]["keywords"]
        pattern = CRITERIA[quan_type]["pattern"]

        if contain_keywords(problem["original_text"], keyword_sets) \
                and re.findall(pattern, problem["equation"]):

            # Add the formula used in the equation to `knowledge`
            # Add the original equation to `history`
            problem["knowledge"].append(FORMULAS[quan_type])
            problem["history"].append(problem["equation"])

            # Make equation substitution
            if quan_type == "volume":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"cuboid_volume({m.group(1)}, {m.group(2)}, {m.group(3)})", problem["equation"])
            elif quan_type == "surface":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"cuboid_surface({m.group(1)}, {m.group(2)}, {m.group(3)})", problem["equation"])
            elif quan_type == "perimeter":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"rectangle_perimeter({m.group(1)}, {m.group(2)})", problem["equation"])
            elif quan_type == "area1":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"rectangle_area({m.group(1)}, {m.group(2)})", problem["equation"])
            elif quan_type == "area2":
                problem["equation"] = re.sub(
                    pattern, lambda m: f"rectangle_area({m.group(1) or m.group(3)}, {m.group(2) or m.group(4)})", problem["equation"])
    return problem


def hard_annotate(problem: Dict):
    """Reannotate the special problems uncovered by the rules.
    """
    if problem["id"] == "1730":
        problem["equation"] = "x=square_area(3)+rectangle_area(3, 4)*4"
        problem["knowledge"] = [
            "(square, side * side = area)",
            "(rectangle, length * width = area)"]
    elif problem["id"] == "9119":
        problem["equation"] = "x=square_area(4)*4"
        problem["knowledge"] = [
            "(square, side * side = area)", ]
    elif problem["id"] == "19048":
        problem["equation"] = "x=rectangle_area(120*(1+(1/3)), 120)"
        problem["knowledge"] = [
            "(rectangle, length * width = area)"]

    return problem

######################################################################
# UTILITY FUNCTIONS
######################################################################


def contain_keywords(text: str,
                     keyword_sets: List[Tuple[str, ...]]
                     ) -> bool:
    """Check if the text contains all the positive keywords and avoids all the negative keywords in at least one input keyword set.

    Negative keywords starts with `^`, e.g., "^xxx".
    See the usage examples below.

        Usage examples
        ----------------------------------
        [Positive keywords]
        # The first case contains all the keywords in the first set
        # The second case contains none of the sets
        >>> text = "I have a few keywords."
        >>> keyword_sets = [("a", "few"), ("a", "number", "of")]
        >>> contain_keywords(text, keyword_sets)
        True
        >>> keyword_sets = [("a", "little"), ("a", "number", "of") ]
        >>> contain_keywords(text, keyword_sets)
        False

        [Negative keywords]
        # fooo in text, thus False
        >>> text = "I have a few keywords and fooo."
        >>> keyword_sets = [("a", "few", "^fooo")]
        >>> contain_keywords(text, keyword_sets)
        False

    """
    for keywords in keyword_sets:
        positive_keywords = [x for x in keywords if not x.startswith("^")]
        negative_keywords = [x.lstrip("^")
                             for x in keywords if x.startswith("^")]

        if all(kw in text for kw in positive_keywords) \
                and not any(kw in text for kw in negative_keywords):
            return True
    return False


def is_square_problem(problem: Dict):
    if "正方" in problem["original_text"]:
        return True
    return False


def is_circle_problem(problem: Dict):
    if "圆" in problem["original_text"]:
        return True
    return False


def is_sphere_problem(problem: Dict):
    if "球" in problem["original_text"] and "半径" in problem["original_text"]:
        return True
    return False


def is_triangle_problem(problem: Dict):
    if "三角" in problem["original_text"]:
        return True
    return False


def is_rectangle_problem(problem: Dict):
    if "长方" in problem["original_text"]:
        return True
    return False
