import numpy as np
from ctypes import c_double
from pathlib import Path

import time
import json
from datetime import datetime

from pbt import *
from sa import *


class Message(object):

    """Wrapper class for a message and its associated attachment."""

    def __init__(self, msg, worker=None):
        self.msg = msg
        self.attachment = worker

    def __eq__(self, string):
        return self.msg == string


def pbt_subroutine(arrs, conns, selection):
    s = selection["subr"]
    if s=="welch":
        pbt_subroutine_welch(arrs, conns, selection)
    elif s=="trunc":
        pbt_subroutine_trunc(arrs, conns, selection)
    else:
        pbt_subroutine_velo(arrs, conns, selection)


def pbt_subroutine_welch(arrs, conns, selection):
    """PBT subprocess: continuously and asynchronously queries worker subprocesses
    for value histories, then chooses whether to replace one with the other
    based on the results of a Welch's t-test."""

    inactivity = 0
    while True:

        a, b = np.random.choice(len(conns), size=2, replace=False)

        if inactivity > selection["inactiv"]:
            conns[a].send(Message("send"))
            template_worker = conns[a].recv()
            conns[a].send(Message("reset", template_worker))
            inactivity = 0
            continue

        conns[a].send(Message("report"))
        a_steps, a_values = conns[a].recv()

        conns[b].send(Message("report"))
        b_steps, b_values = conns[b].recv()

        pval, mudiff = welchs(a_values, b_values)

        if pval < selection["p"]:
            inactivity = 0

            if mudiff > 0 and a_steps >= selection["n protected"]:  # > means we are minimizing
                better_worker_index = a
                worse_worker_index = b
            elif mudiff < 0 and b_steps >= selection["n protected"]:
                better_worker_index = b
                worse_worker_index = a
            else:
                inactivity += 1
                continue

            conns[better_worker_index].send(Message("send"))
            template_worker = conns[better_worker_index].recv()
            conns[worse_worker_index].send(Message("reset", template_worker))

        else:
            inactivity += 1


def pbt_subroutine_velo(arrs, conns, selection):
    """PBT subprocess: continuously and asynchronously queries worker subprocesses
    for value histories, then chooses whether to replace one with the other
    based on the expected value after some time."""

    inactivity = 0
    while True:

        a, b = np.random.choice(len(conns), size=2, replace=False)

        if inactivity > selection["inactiv"]:
            conns[a].send(Message("send"))
            template_worker = conns[a].recv()
            conns[a].send(Message("reset", template_worker))
            inactivity = 0
            continue

        conns[a].send(Message("report"))
        a_steps, a_values = conns[a].recv()
        conns[b].send(Message("report"))
        b_steps, b_values = conns[b].recv()

        winner = velo(a_values, b_values, 1-selection["p"])
        if winner is None:
            inactivity += 1
            continue

        if winner==0 and b_steps >= selection["n protected"]:
            better_worker_index = a
            worse_worker_index = b
        elif winner==1 and a_steps >= selection["n protected"]:
            better_worker_index = b
            worse_worker_index = a
        else:
            inactivity += 1
            continue

        conns[better_worker_index].send(Message("send"))
        template_worker = conns[better_worker_index].recv()
        conns[worse_worker_index].send(Message("reset", template_worker))


def pbt_subroutine_trunc(arrs, conns, selection):
    """PBT subprocess: continuously and asynchronously queries worker subprocesses
    for value histories, then chooses whether to replace one with the other
    based on their position in a ranking of workers."""

    while True:
        summary = assemble_summary_array(arrs, len(conns), csv_header)

        a = np.random.choice(len(conns))

        low, high = get_extremes(summary['Value'].values, pctg=selection["trunc"])

        if a in low:
            b = np.random.choice(high)
        else:
            continue

        conns[b].send(Message("report"))
        b_steps, _ = conns[b].recv()

        if b_steps >= selection["n protected"]:
            conns[b].send(Message("send"))
            template_worker = conns[b].recv()
            conns[a].send(Message("reset", template_worker))


def worker_subroutine(seed, i, instance, a, conn,
                      inits, perturb_scales, history_horizon,
                      item_values, item_weights, knapsacks, optimal_value):
    """A spawned subprocess will execute this code. All connections
    (from here to PBT and spawner processes) remain open
    for the duration of the execution."""

    worker = SAWorker(seed, i, instance, inits, history_horizon,
                      item_values, item_weights, knapsacks)

    while True:
        worker.step()
        while conn.poll():
            msg = conn.recv()
            if msg == "report":
                conn.send((worker.n_steps, worker.last_values))
            elif msg == "send":
                conn.send(worker)
            elif msg == "reset":
                template = msg.attachment
                worker = exploit(template, worker)
                worker = explore(worker, perturb_scales)
        a[:] = worker.summary_vector()


def scribe_subroutine(arrs, params, delay=1):
    csv_header = "Time,Worker,Age,Value,Temperature,Cooling rate,Mutation prob."
    datestr = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    records_dir = Path('.') / 'records' / datestr
    records_dir.mkdir(exist_ok=True)
    records_file = records_dir / "history.csv"
    info_file = records_dir / "info.json"
    with info_file.open('w') as f:
        f.write(json.dumps(params, sort_keys=True, indent=4))
    with records_file.open('w') as f:
        f.write(csv_header+'\n')
        t0 = time.time()
        while True:
            elapsed_time = (time.time()-t0)/60
            df = assemble_summary_array(arrs, elapsed_time, csv_header)
            print(df[['Value','Temperature','Cooling rate','Mutation prob.','Age']].sort_values(by=['Value','Temperature'], ascending=False))
            f.write(make_csv_lines(df))
            time.sleep(delay)
