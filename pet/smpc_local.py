import numpy as np


def make_secure(params: float or int or dict or list, n: int, r: int = 1000000, operation: str = 'add') -> list:
    """
    Create the shards to distribute across the various clients.
    :param params: A float, int, list of ints or floats, or dictionary with string keys and float,
    int or list of ints or floats values.
    :param n: int number of shards to generate
    :param r: int range for the produced noise
    :param operation: 'add' or 'multiply' operation
    :return: noisy parameters ps with the same structure as params
    """
    if operation != 'add':
        raise Exception('Operation is not allowed here.')
    ps = []
    if type(params) == dict:
        for i in range(n):
            ps.append({})
        for key in params.keys():
            pd = make_secure(params[key], n, r)
            for i in range(len(ps)):
                ps[i][key] = pd[i]
    elif type(params) == list:
        for i in range(n):
            ps.append([])
        for pj in params:
            pd = make_secure(pj, n, r)
            for i in range(len(ps)):
                ps[i].append(pd[i])
    elif type(params) == float or int:
        rs = 0.0
        for i in range(n - 1):
            r = skewed_rand(r)
            ps.append(r)
            rs += r

        ps.append(params - rs)
    else:
        raise Exception("This type is not supported. Please only create shards for float, lists or dicts.")

    return ps


def skewed_rand(r: float) -> float:
    """
    :param r: float r
    :return: random numbers that are equally likely inside the interval (1/r, 1] and [1, r)
    """
    return np.random.randint(-r, high=r, dtype=int)


# transforms a float into an int
def to_int(f: float, exp: int) -> int:
    return int(f * np.exp(exp))


# transforms an int into a float
def to_float(i: int, exp: int) -> float:
    return float(i) / np.exp(exp)
