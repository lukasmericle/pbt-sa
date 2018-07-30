from scipy.stats import ttest_ind, linregress, theilslopes
from numpy import hstack, vstack, zeros, arange, newaxis
from pandas import DataFrame


def exploit(from_worker, to_worker):
    """`exploit` step in PBT."""
    to_worker.reset(from_worker)
    return to_worker


def explore(worker, scales):
    """`explore` step in PBT.
    Comment out as needed."""
    worker.perturb_temp(scales["temperature"])
    worker.perturb_cooling_rate(scales["cooling rate"])
    worker.perturb_p_mutations(scales["p mutations"])
    return worker


def welchs(a1, a2):
    """Welch's t-test."""
    pval = ttest_ind(a1, a2, equal_var=False)[1]
    mudiff = sum(a1)/len(a1) - sum(a2)/len(a2)
    return pval, mudiff


def velo(a1, a2, alpha):
    """Measures improvement velocity of workers. Returns index of the one which
    is expected to be better after a given time."""
    n = len(a1) + len(a2) - 2
    medslope1, medintercept1, lo_slope1, up_slope1 = theilslopes(a1, alpha=alpha)
    medslope2, medintercept2, lo_slope2, up_slope2 = theilslopes(a2, alpha=alpha)
    anchor1 = medslope1 * (len(a1)-1)/2 + medintercept1
    anchor2 = medslope2 * (len(a2)-1)/2 + medintercept2
    y1lo = lo_slope1 * (n - (len(a1)-1)/2) + anchor1  # extrapolate out and see
    y1hi = up_slope1 * (n - (len(a1)-1)/2) + anchor1  # which leads to higher score
    y2lo = lo_slope2 * (n - (len(a2)-1)/2) + anchor2  # at present rate of
    y2hi = up_slope2 * (n - (len(a2)-1)/2) + anchor2  # improvement
    if y1lo > y2hi:
        return 0
    if y2lo > y1hi:
        return 1
    return None


def get_extremes(arr, pctg=0.2):
    """Get the top and bottom `pctg`% of the array."""
    cutoff = max(1, int(pctg*len(arr))+1)
    sort_order = [x for y,x in sorted(zip(arr, range(len(arr))))]
    return sort_order[:cutoff], sort_order[-cutoff:]


def assemble_summary_array(arrs, time, csv_header):
    arr = vstack([a.get_obj() for a in arrs]).reshape((len(arrs), -1))
    arr = hstack([zeros(len(arrs))[:,newaxis] + time, arange(len(arrs))[:,newaxis], arr])
    df = DataFrame(arr, columns=csv_header.split(","))
    return df


def make_csv_lines(df):
    out = ""
    for index, row in df.iterrows():
        out += ",".join(["{:.6f}".format(val) for val in row.values])+"\n"
    return out
