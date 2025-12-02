"""
Math verify stub for MMLU evaluation
Simplified version for answer verification
"""

def parse(response: str):
    """Parse answer from response."""
    import re
    
    if not response:
        return []
    
    response = response.strip()
    
    # Try to find boxed answer
    boxed_match = re.search(r'\\boxed\{([A-Za-z])\}', response)
    if boxed_match:
        return [boxed_match.group(1).upper()]
    
    # Try "The answer is (X)"
    answer_match = re.search(r'The answer is\s*\(?([A-Z])\)?', response, re.IGNORECASE)
    if answer_match:
        return [answer_match.group(1).upper()]
    
    # Try "Answer: X"
    answer_match2 = re.search(r'Answer:\s*([A-Z])', response, re.IGNORECASE)
    if answer_match2:
        return [answer_match2.group(1).upper()]
    
    # Last resort: find any single capital letter
    letter_match = re.search(r'\b([A-D])\b', response)
    if letter_match:
        return [letter_match.group(1)]
    
    return []


def verify(gold, parsed):
    """Verify if parsed answer matches gold."""
    if not parsed or parsed is None:
        return False
    
    if isinstance(parsed, list) and len(parsed) > 0:
        parsed_ans = parsed[0].upper() if isinstance(parsed[0], str) else str(parsed[0]).upper()
    else:
        parsed_ans = str(parsed).upper()
    
    gold_ans = gold.upper() if isinstance(gold, str) else str(gold).upper()
    
    return parsed_ans == gold_ans