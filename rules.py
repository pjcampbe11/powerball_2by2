import yaml
import operator

OPS = {
    ">": operator.gt, ">=": operator.ge,
    "<": operator.lt, "<=": operator.le,
    "==": operator.eq
}

def load_rules(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("rules", [])

def evaluate_rules(color: str, pair, facts: dict, rules: list):
    hits = []
    for rule in rules:
        matched = True
        for field, condition in rule.get("when", {}).items():
            op, threshold = condition.split()
            threshold = float(threshold)
            if field not in facts:
                matched = False
                break
            if op not in OPS:
                matched = False
                break
            if not OPS[op](float(facts[field]), threshold):
                matched = False
                break

        if matched:
            hits.append({
                "color": color,
                "pair": pair,
                "rule": rule.get("name", "UNKNOWN"),
                "severity": rule.get("severity", "unknown"),
                "frequency": facts.get("frequency"),
                "weighted_score": facts.get("weighted_score"),
                "chi_square": facts.get("chi_square"),
            })
    return hits
