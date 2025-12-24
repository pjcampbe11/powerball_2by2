import math

EPS = 1e-12

def shannon_entropy(probs):
    h = 0.0
    for p in probs:
        if p > 0:
            h -= p * math.log(p, 2)
    return h

def kl_divergence(p, q):
    d = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, EPS)
        qi = max(qi, EPS)
        d += pi * math.log(pi / qi, 2)
    return d

def js_divergence(p, q):
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def g_test_stat(observed, expected):
    g = 0.0
    for o, e in zip(observed, expected):
        if o > 0 and e > 0:
            g += 2.0 * o * math.log(o / e)
    return g

def chi_square_stat(observed, expected):
    x2 = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            x2 += (o - e) ** 2 / e
    return x2

def approx_chi_square_pvalue(x2, df):
    if df <= 0:
        return 1.0
    a = 1 - 2/(9*df)
    b = math.sqrt(2/(9*df))
    z = ((x2/df) ** (1/3) - a) / b
    return 0.5 * math.erfc(z / math.sqrt(2))

def window_stats(counts_dict, support_keys):
    obs = [counts_dict.get(k, 0) for k in support_keys]
    total = sum(obs)
    if total == 0:
        return {"total": 0, "entropy_bits": 0, "kl_to_uniform_bits": 0, "js_to_uniform_bits": 0,
                "chi2": 0, "g_test": 0, "chi2_p_approx": 1.0}

    k = len(support_keys)
    expected = [total / k] * k
    p = [o / total for o in obs]
    q = [1.0 / k] * k

    H = shannon_entropy(p)
    KL = kl_divergence(p, q)
    JS = js_divergence(p, q)
    x2 = chi_square_stat(obs, expected)
    g = g_test_stat(obs, expected)
    pval = approx_chi_square_pvalue(x2, df=k-1)

    return {"total": total, "entropy_bits": H, "kl_to_uniform_bits": KL, "js_to_uniform_bits": JS,
            "chi2": x2, "g_test": g, "chi2_p_approx": pval}
