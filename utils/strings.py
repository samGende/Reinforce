import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
EOS_RE = re.compile(f"<|end_of_text|>")

INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def check_eos(completion):
    match = EOS_RE.search(completion)
    return match != None
