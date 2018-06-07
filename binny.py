"""
Binning functions
"""

import numpy as np
import pandas as pd
from .general_helpers import (
    get_limits, parse_unit, replacement_map, 
    is_number_like, is_string_like, is_map_like, is_hashable,
    getLogger
)

# sometimes we want to use a string to specify an aggregation function 
# that pandas doesn't know
def se(x):
    """Return standard error of x."""
    return np.std(x) / np.sqrt(len(x))
    
func_replacements = {
    'se': se, # standard error 
}


# Helpers to specify bins
def bin_edges(x, bins, lim=None, space='lin', right=False):
    """Return lin- or log-spaced bin edges. See: bin_df
    """
    x = np.asanyarray(x) # safety first
    x = x[np.isfinite(x)]
    
    if space=='log':
        # complicated: maybe we got some negative x
        # then we have to split into positive and negative
        # maybe we just have a zero to treat separately
        xmin, xmax = get_limits(x, lim)
        if xmin < 0 and xmax > 0:
            b = int(np.ceil(bins / 2.))
            # not enough edges without zero?
            if 2 * b < bins + 1:
                bin_zero = [0]
            else:
                bin_zero = []
        elif (xmin == 0 and not right) or (xmax == 0 and right):
            b = bins
            bin_zero = [0]
        else:
            b = bins + 1
            bin_zero = []

        bin_edges_p = []
        bin_edges_n = []
        
        # symmetrical binning?
        if xmin == -xmax:
            symmetric = True
            xnear     = np.nanmin(np.abs(x[x!=0]))
        else:
            symmetric = False

        # now to the half-spaces (if they exist)
        if xmax > 0:
            # positive part of [xmin, xmax]
            if symmetric:
                pmin = xnear
                pmax = xmax
            else:
                xp  = x[ (x > max(0,xmin)) & (x < xmax)]
                pmin, pmax = get_limits(xp, ('near','far'))
            bin_edges_p = np.logspace(
                np.log10(pmin) ,np.log10(pmax), b
            )
        if xmin < 0:
            # negative part of [xmin, xmax]
            if symmetric:
                # here everything is flipped
                # - sign is added to logspace below
                nmax = xnear
                nmin = -xmin
            else:
                xn  = - x[(x < min(0, xmax)) & (x > xmin)]
                nmax, nmin = get_limits(xn, ('near','far'))
            bin_edges_n = - np.logspace(
                np.log10(nmin), np.log10(nmax), b
            )#[::-1] # logspace can go from larg to small!
        # all edges
        bin_edges = np.r_[bin_edges_n, bin_zero, bin_edges_p]
    # Ok, the rest is easy:
    elif "q" in space:
        # linear in quantiles
        if lim:
            # limits take time, do only if necessary
            xmin, xmax = get_limits(x, lim)
            x = x[(xmin < x) & (x < xmax)]
        l, u = parse_unit(space)
        p_low  = 100 * (1 - l)
        p_high = 100 * (l)
        assert p_high > 50, "Pass higher quantile as kwarg space!"
        bin_edges = np.percentile(
            x,
            np.linspace(p_low, p_high, bins + 1) 
        )
    elif space == 'lin':
        # just linear
        xmin, xmax = get_limits(x, lim)
        bin_edges  = np.linspace(xmin, xmax, bins + 1)
    else:
        raise ValueError("Space must be 'lin', 'log', or 'q'.")
    
    # only return unique bins, particularly important for q
    bin_edges = np.unique(bin_edges)
    
    # avoid single data point in extra bin
    if right:
        bin_edges[0]  -= 10**-10
    else:
        bin_edges[-1] += 10**-10
        
    # done
    return bin_edges

def bin_centers_from_edges(edges, keep_outer_edges=False):
    """Return bin centers calculated from edges.
    See also: bin_edges."""
    edges   = np.asanyarray(edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    if keep_outer_edges:
        centers = np.r_[min(edges), centers, max(edges)]
    return centers

# do the actual binning
def bin_df(
        df, 
        bins             = 100,
        by               = None, 
        lim              = None,
        space            = 'lin',
        bin_col_func     = 'mean',
        bin_pos_func     = 'mean',
        right            = False,
        drop_edges       = False,
        return_log       = False,
        return_bin_edges = False,
        rename_cols      = None
    ):
    """[Log-]Bin an array [by another array] or DataFrame [by column(name)].
    Return a pandas DataFrame.
    
    Parameters:
    ===========
    
    df: pd.DataFrame, np.array, ...
        The data to be logbinned
    bins: int [optional]
        The number of bins. Empty bins are not returned. Returned bin
        positions correspond to the mean of the positions of data points
        in the bin.
    by: None, str, ... [optional]
        Default: None = Use the (existing) index 
        Otherwise, this parameter will be passed to the pandas DataFrame 
        groupby method.
    lim: None, int, str, list-like [optional]
        Definition of the limits we want. Default: None
        see pna.general_helpers.get_limits
    space: str [optional]
        Bin spacing.
        'lin': linear (default). 'log': logarithmic, 'q': quantile.
    bin_col_func: str, ufunc, list [optional]
        How to calculate the value of each bin after grouping. 
        See pandas' agg.
        Examples:
        'median', lambda x: np.exp(np.mean(np.log(x))), ...
        Named functions give correctly named columns.
        [f1, f2] applies f1 and f2 to each column in df; result has a multi
        index for the columns. 
        Default: 'mean'.
    bin_pos: str, ufunc, None [optional]
        How to calculate the bin centers.
        E.g. 'mean', np.mean use the mean positions of the data points
        in each bin. 
        If bin_pos evaluats false ('', None, False), the centers the
        bins are used ignoring how the data is distributed in each bin.
    right: bool [optional]
        Indicating whether the intervals include the right or the left bin
        edge. Default: False. Passed to numpy's digitize function.
    drop_edges: bool [optional]
        If true, (bins - 2) bins are used within the range acoording to lim.
        The two outermost bins aggregate all data outside of that range.
        Omitted if no bins would be left after dropping.
        Default: False
    return_log: bool [optional]
        Whether to return the logarithms of bin values and positions.
        Default: False.
    return_bin_edges: bool [optional]
        Return the bin edges as the second parameter. Default: False
    Notes: 
    ======
    
    bin_col_func and bin_cos_func are compared against func_replacements.
    Can be used to calculate conditional expectation values, see 
    aggregate_impact
    """
    
    # prep dataframe and axes ##################
    # do we need to copy?
    copy_df = True
    # y-axis
    if type(df) is not pd.DataFrame:
        df = pd.DataFrame(df)
        copy_df = False
    
    # x-axis
    by_key  = None
    by_name = None
    if type(by) == type(None):
        # default: use index
        x = df.index.values
        if copy_df:
            df = df.copy()
        if df.index.name:
            # if index has a name, use it
            by_name = df.index.name
    elif is_hashable(by) and by in df:
        # by is an existing column
        x = df[by]
        # therefore by is also a proper key
        by_name = by
        by_key  = by
    elif len(by) == len(df):
        # try to use what user passed (e.g. Series, array)
        if copy_df:
            df = df.copy()
        x = by
        if hasattr(by, 'name') and by.name:
            # If by has a name (e.g. is Series), use it
            by_name = by.name
    else:
        raise ValueError("I don't understand the value of kwarg 'by'.")
    
    # check on binning function for bin positions
    # to calc weighted bin-centers
    bin_pos_func = replacement_map(bin_pos_func, func_replacements)
    if hasattr(bin_pos_func, '__name__'):
        bin_pos_func_name = bin_pos_func.__name__
    else:
        bin_pos_func_name = str(bin_pos_func)
    
    # do we still need a column-compatible key for by?
    if by_key is None:
        # what about a name?
        if by_name is None: by_name = 'x' # generic
        # we still need a proper key depending on the column depth
        clevels = df.columns.nlevels
        if clevels > 1:
            # we hae to assume some reasonable column structure...
            ## first level: by, second level: binning function for by,
            ## other levels: ''
            by_key = (by_name, bin_pos_func_name) + ("",) * (clevels-2)
        else:
            by_key = by_name
    
    if (
            # do we want to bin the 'by'-values?
            bin_pos_func
            # ok, is by missing from df's cols?
            and not (
                    # if by_key and df have the same nlevels, 
                    # find by_key:
                    by_key in df
                    # this will work for multi-columns 
                    # and str as by_key
                or by_key in df.columns.get_level_values(0)
            )
        ):
        # we have to add by as col
        df[by_key] = x
    
    # bin edges ################################  
    if hasattr(bins, '__len__'):
        # we got bin edges
        be = bins
    else:
        # digitize two additional "out of range" bins
        # if we returned those, we would return two
        # more bins than requested
        if not drop_edges: bins -= 2
        be = bin_edges(x, bins, lim=lim, space=space, right=right)
    
    # binning functions ################################  
    bin_col_func = replacement_map(bin_col_func, func_replacements)
    if bin_col_func and not is_map_like(bin_col_func):
        # every column gets the same binning,
        # except by (should now be a string, gets special treatment below)
        fn = { 
            c: bin_col_func for c in df.columns if c != by_key
        }
    else:
        # (hopefully) the user specified everything
        fn = bin_col_func
    
    # the extra column for bin centers
    if bin_pos_func:
        fn[by_key] = bin_pos_func
        
    # the actual binning ############################
    try:
        di = np.digitize(np.asarray(x).flatten(), be, right=right)
    except Exception as e:
        # debug
        info = (
              "x: min  =" + str(np.min(x)) 
            + "max = " + str(np.max(x))
            + "len = " + len(x)
            + "\nbins: " + str(np.min(be)) 
            + "max = " + str(np.max(be))
            + "len = " + len(np.unique(be))
        )
        getLogger(__name__).error(info, e)
        raise e
    binned = (
        df
        .groupby(di)
    ).agg(
        fn
    )
    # finalise ############################
    if return_log:
        binned = np.log10(binned)
        
    if drop_edges and len(binned) > 2:
        # return only values between binedges
        binned = binned.iloc[1:-1]
    
    ## bin centers as index
    if bin_pos_func:
        # if we had several binning functions, binning has
        # added an additional level to columns!
        if (binned.columns.nlevels == 1):
            binned_by_key = by_key
        else:
            if is_string_like(by_key):
                binned_by_key = (by_key,) + (bin_pos_func_name,)
            else:
                binned_by_key = by_key + (bin_pos_func_name,)

        # weighted center
        # extra steps to make sure index is one-dimensional!
        ix = binned[by_key]
        binned.set_index(binned_by_key, inplace=True)
        # try to get rid of unused labels in columns
        binned.columns    = binned.columns.drop_duplicates()
        binned.index.name = by_name
    else:
        # a-priori bin-centers
        # if we don't drop the edges, we want the
        # centers + the outer edges.
        # else, we drop the bins number 0 and bins+1 - hence
        # we have to shift the index by one.
        bc = bin_centers_from_edges(be, keep_outer_edges=not drop_edges)        
        binned.index = bc[binned.index.values.flatten() - drop_edges]
    
    if rename_cols:
        binned.columns = rename_cols
        
    # done
    if return_bin_edges:
        return binned, be
    else:
        return binned

