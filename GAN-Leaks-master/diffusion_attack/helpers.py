from collections import OrderedDict


def fix_legacy_dict(d):
    keys = list(d.keys())
    if "model" in keys:
        d = d["model"]
    if "state_dict" in keys:
        d = d["state_dict"]
    keys = list(d.keys())
    # remove multi-gpu module.
    if "module." in keys[1]:
        d = remove_module(d)
    return d


def remove_module(d):
    return OrderedDict({(k[len("module.") :], v) for (k, v) in d.items()})