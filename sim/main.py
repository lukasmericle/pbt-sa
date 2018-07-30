import os
import time
import json
import multiprocessing as mp

from ctypes import c_double

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ksio import *
from subr import *


def default_cfg():

    params = {
        "time limit" : 10,
        "n workers"  : 50,
        "baseline"   : False,
        "filename"   : "./data/sac94/weing/weing8.dat",
        "inits" : {
            "temperature"  : ('unilog', 3, 6),
            "cooling rate" : ('unilog', -4, -2),
            "p mutations"  : ('expit', 0.01, 0.99),
        },
        "scales" : {
            "temperature"  : 0.05,
            "cooling rate" : 0.05,
            "p mutations"  : 0.05,
        },
        "selection" : {
            "subr"        : "velo",
            "p"           : 0.01,
            "trunc"       : 0.05,
            "inactiv"     : 50,
            "n protected" : 50,
        },
        "horizon" : 50,
    }

    with open("./cfg.json",'w') as f:
        f.write(json.dumps(params, sort_keys=True, indent=4))

    return params


if __name__=="__main__":

    try:
        with open("./cfg.json", 'r') as f:
            params = json.load(f)
    except:
        params = default_cfg()
    params["instance"] = params["filename"].split("/")[-1].split(".")[0]

    item_values, item_weights, knapsacks, optimal_value = read_file(params["filename"])

    workers = []
    parent_conns = []
    a = [mp.Array(c_double, 5) for _ in range(params["n workers"])]
    for i in range(params["n workers"]):
        parent_conn, child_conn = mp.Pipe()
        parent_conns.append(parent_conn)
        seed = np.random.randint(2**32 - 1)
        args = (seed, i, params["instance"], a[i], child_conn,
                params["inits"], params["scales"], params["horizon"],
                item_values, item_weights, knapsacks, optimal_value)
        worker = mp.Process(target=worker_subroutine, args=args, name="Worker {}".format(i))
        workers.append(worker)
        workers[-1].start()

    args = (a, params)
    scribe = mp.Process(target=scribe_subroutine, args=args, name="Scribe")
    scribe.start()

    if (not params["baseline"]) and (params["n workers"] > 1):
        args = (a, parent_conns, params["selection"])
        pbt = mp.Process(target=pbt_subroutine, args=args, name="PBT")
        pbt.start()

    time.sleep(params["time limit"]*60)  # TODO: replace with less crude method, i.e., tell processes to shut themselves down and then release resources once that is complete

    scribe.terminate()
    if (not params["baseline"]) and (params["n workers"] > 1):
        pbt.terminate()
    for worker in workers:
        worker.terminate()
