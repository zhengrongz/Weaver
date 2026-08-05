import re

def format_reward(solution_str: str) -> float:
    pattern = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL)
    match = re.fullmatch(pattern, solution_str)
    return 1.0 if match else 0.0
    

def acc_reward(solution_str: str, ground_truth: str) -> float:
    answer = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not answer:
        return 0.0
    if ground_truth.startswith('('):
        ground_truth = ground_truth[1].upper()
    answer = re.search(r"[A-F]", answer.group(1))
    if not answer:
        return 0.0
    answer = answer.group(0)
    return 1.0 if answer == ground_truth else 0.0
    
def compute_score(data_source, solution_str: str, ground_truth: str, extra_info=None) -> float:
    acc = acc_reward(solution_str, ground_truth)
    format = format_reward(solution_str)
    score = 0.1 * format + 0.9 * acc
    return {
        'score': score,
        'acc': acc,
        'format': format,
    }