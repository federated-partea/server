

def apply_operation(params):
    s = []
    for i in range(len(params[0])):
        if isinstance(params[0][i], float) or isinstance(params[0][i], int):
            p = 0.0
            for param in params:
                p += param[i]
            s.append(p)
        elif isinstance(params[0][i], list):
            ps = []
            for param in params:
                ps.append(param[i])
            p = apply_operation(ps)
            s.append(p)
        elif isinstance(params[0][i], dict):
            p = {}
            for key in params[0][i].keys():
                ps = []
                for param in params:
                    try:
                        ps.append([param[i][key]])
                    except KeyError:
                        ps.append([0.0])
                p[key] = apply_operation(ps)
                if len(p[key]) == 1:
                    p[key] = p[key][0]
            s.append(p)

    return s


def sum_dicts(dicts: [dict]):
    return {k: sum(d[k] for d in dicts) for k in dicts[0]}


def aggregate_smpc(params, exp):
    """
    Aggregates parameters into a new parameter struct based on the specified operation
    :param params:
    :param operation:
    :return:
    """
    if len(params) > 1:
        agg = apply_operation(params=params)
    else:
        while isinstance(params[0], list):
            params = params[0]
        return params[0]
    while isinstance(agg[0], list):
        agg = agg[0]

    result = from_int(agg[0], exp=exp)
    return result


def distribute_smpc(data: dict):
    distribute_dict = {}
    for mem in data.keys():
        client_dist = []
        for key2 in data.keys():
            client_dist.append([data[key2][mem]])
        distribute_dict[mem] = client_dist
    return distribute_dict


def from_int(params: int or float or dict, exp: int) -> int or float or dict:
    p = None
    if type(params) == float or type(params) == int:
        p = float(params) / 10 ** exp
    elif type(params) == list:
        p = []
        for param in params:
            p.append(from_int(param, exp))
    elif type(params) == dict:
        p = {}
        for key in params.keys():
            p[key] = from_int(params[key], exp)

    return p
