"""

circumference_diameter

"""

NUM_OF_ARGS = {
    "square_perimeter": 1,
    "square_area": 1,
    "cubic_volume": 1,
    "circle_area": 1,
    "circumference_radius": 1,
    "circumference_diameter": 1,
    "triangle_area": 2,
    "rectangle_area": 2,
    "rectangle_perimeter": 2,
    "cuboid_volume": 3,
    "cuboid_surface": 3,
}

FUNC_NAMES = list(NUM_OF_ARGS.keys())


def evaluate_custom_function(func_name, args):
    """
        Example usage
        ------------------------------
        >>> evaluate_custom_function("square_perimeter", [5])
        20
        >>> round(evaluate_custom_function("circle_area", [6]), 2)
        113.04
    """
    assert func_name in FUNC_NAMES
    assert len(args) == NUM_OF_ARGS[func_name]

    if func_name == "square_perimeter":
        return args[0] * 4
    elif func_name == "square_area":
        return args[0] * args[0]
    elif func_name == "cubic_volume":
        return args[0] * args[0] * args[0]
    elif func_name == "circle_area":
        return 3.14 * args[0] * args[0]
    elif func_name == "circumference_radius":
        return 2 * 3.14 * args[0]
    elif func_name == "circumference_diameter":
        return 3.14 * args[0]
    elif func_name == "triangle_area":
        return args[0] * args[1] / 2
    elif func_name == "rectangle_area":
        return args[0] * args[1]
    elif func_name == "rectangle_perimeter":
        return 2 * (args[0]+args[1])
    elif func_name == "cuboid_volume":
        return args[0] * args[1] * args[2]
    elif func_name == "cuboid_surface":
        return 2 * (args[0]*args[1] + args[0]*args[2] + args[1]*args[2])
