from math import ceil,sqrt

def L1(n):
    denominator = ceil(sqrt(n)) - 1.0
    return 1.0 / denominator

def L2(n):
    denominator = ceil(sqrt(n + 1.0)) - 3.0 + sqrt(2.0 + sqrt(3.0))
    return 1.0 / denominator

def L3a(n):
    denominator = ceil(sqrt(n + 2.0)) - 2.0 + (0.5 * sqrt(3.0))
    return 1.0 / denominator

def L3b(n):
    denominator = ceil(sqrt(n + 2.0)) - 5.0 + (2.0 * sqrt(2.0 + sqrt(3.0)))
    return 1.0 / denominator

def L4(n):
    m = ceil(sqrt(2.0 * n))
    for k in range(1,m+1):
        if n == k * (k + 1.0):
           numerator = k**2.0 - k - sqrt(2.0 * k) 
           if numerator == 0:
               return 0
           denominator = k**3.0 - (2.0 * k**2.0)
           return numerator / denominator
    return 0

def L5(n):
    m = ceil(sqrt(4.0 * n))
    for p in range(1,m+1):
        for q in range(1,m+1):
            if (p**2 < 3 * q**2) and (q**2 < 3 * p**2):
                if n == ceil(((p + 1) * (q + 1)) / 2):
                    return sqrt((1/p**2) + (1/q**2))
    return 0

def approximate_maxmin_distance(n):
    if n == 3:
        return sqrt(6) - sqrt(2)
    else:
        return max(L1(n),L2(n),L3a(n),L3b(n),L4(n),L5(n))