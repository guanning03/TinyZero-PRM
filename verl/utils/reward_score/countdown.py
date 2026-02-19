import re
import ast
import operator
import warnings


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def _safe_eval_node(node):
    """Recursively evaluate an AST node, supporting only +, -, *, / on numbers.

    This replaces the previous ``eval()``-based implementation which could hang
    indefinitely when the model generated expressions containing the ``**``
    (power) operator — e.g. ``480**782**576`` produces an astronomically large
    number that Python will spend forever computing.

    By walking the AST ourselves we guarantee:
    * Only ``+  -  *  /`` (and unary ``-``, ``+``) are allowed.
    * ``**`` (Pow) and every other operator / construct is rejected immediately.
    * No risk of code-injection via ``eval()``.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    elif isinstance(node, ast.Num):          # Python 3.7 compat
        return node.n
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        op_type = type(node.op)
        if op_type is ast.Add:
            return left + right
        elif op_type is ast.Sub:
            return left - right
        elif op_type is ast.Mult:
            return left * right
        elif op_type is ast.Div:
            if right == 0:
                raise ValueError("Division by zero")
            return left / right
        else:
            raise ValueError(f"Unsupported binary operator: {ast.dump(node.op)}")
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.UAdd):
            return +operand
        else:
            raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
    else:
        raise ValueError(f"Unsupported AST node: {ast.dump(node)}")


def evaluate_equation(equation_str):
    """Safely evaluate an arithmetic equation using AST parsing.

    Only supports ``+``, ``-``, ``*``, ``/`` on numeric literals.
    Returns ``None`` for any invalid / unsupported expression.
    """
    try:
        # Quick sanity: reject extremely long strings
        if len(equation_str) > 500:
            return None
        tree = ast.parse(equation_str, mode='eval')
        return _safe_eval_node(tree)
    except Exception:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)

    if equation is None:
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            return score
        else:
            return format_score
    except:
        return format_score 