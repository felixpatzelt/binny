"""
Helpers, maybe factor to separate package later.
"""
import numpy as np
from matplotlib.cbook import is_numlike, is_string_like
from collections import Mapping
#from copy import deepcopy
try:
    from progress import getLogger
except ImportError:
    from logging import getLogger

##########################################################################
# Numbers, units, extents, ...
##########################################################################

def is_number_like(obj):
    """Return True if obj looks like a SINGLE number."""
    try:
        obj = obj + 1  # might still be an array!
        obj = float(obj)
    except:
        return False
    # all cool
    return True
    
def is_map_like(obj):
    """Return True if obj is be derived from collection.Mapping)"""
    return isinstance(obj, Mapping)

def is_hashable(obj):
    """Return True if obj is hashable, else return False."""
    try:
        hash(obj)
        return True
    except:
        return False
        
#def re_sub(string, patterns ):
#    """Apply a list of (regex) substitutions to string and return it."""
#    for p, r in patterns:
#        pc = re.compile(p)
#        string = pc.sub(r, string)
#    return string

def replacement_map(x, replacements):
    """check replacemets for x or elements in x
    
    Should handle dicts, lists, tuples, strings, functions, and similar.
    """
    if hasattr(x, 'iteritems'): # e.g. dict
        # print type(x), x, 'is map like'
        res = {
            k: replacement_map(v, replacements)
            for k, v in x.iteritems()
        }
    elif (
                    hasattr(x, '__getitem__') # iterable
            #    not hasattr(x, '__call__')    # no func
            and not is_string_like(x)         # no str
            and not is_number_like(x)         # no num
        ):
        # print type(x), x, 'is sequence but no string'
        res = [
            replacement_map(i, replacements)
            for i in x
        ]
    else:
        # print type(x), x, 'is key like'
        if x in replacements:
            res = replacements[x]
        else:
            res = x
    return res

def parse_unit(x):
    """Return number, 'unit' for x=str(number)+'unit'."""
    if is_numlike(x):
        return x, None
    else:
        for i, xi in enumerate(x+" "):
            if not xi.isdigit() and not xi == '.':
                break # we only want the right i
        number = x[:i]
        unit   = x[i:].strip()
        if unit == '':
            unit = None
        # return result
        if number.isdigit():
            return int(number), unit
        elif number == "":
            return 1, unit
        else:
            return float(number), unit

def _apply_limit_abs_unit(x, lim, unit):
    """Return one limit with applied unit(abs(x)). See get_limits."""
    if unit is None:
        return lim
    unit = unit.lower()
    if unit == 'near':
        return lim * np.nanmin(np.abs(x))
    if unit == 'far':
        return lim * np.nanmax(np.abs(x))
    elif unit == 'median':
        return lim * np.nanmedian(np.abs(x))
    elif unit == 'mean':
        return lim * np.nanmean(np.abs(x))
    else:
        raise ValueError, "Unknown unit %s"%unit
        
def _parse_limit(lim):
    """Returns limit_number, limit_unit. See get_limits."""
    if lim is None:
        return (1, 'far')
    elif is_string_like(lim):
        return parse_unit(lim)
    elif is_number_like(lim):
        return (lim, None)
    else:
        raise ValueError, "Can't parse limit: %s"%str(lim)

def get_limits(x, lim=None):
    """Return the range of values in x.
    
    Parameters:
    ===========
    x: array like
        The values to analyse
    lim: None, int, str, list-like
        Definition of the limits we want. Default: None
        A single value means symmetric limits depending on abs(x), except
        when x is strictly positive (negative) where one limit will be 
        zero.
        Two values (tuple, etc) means asymmetric limits.
        lim can be a string with a unit:
            'mean':   mean(abs(x)))
            'median': median(abs(x))
            'far':    max(abs(x))
            'near':   min(abs(x))
        For asymmetric limits, the unit is calculated for positive and
        negative values separately (if there are any).
        The unit string can include a multiplier, i.e. '5 unit'.
    
    Examples:
    =========
    >>> get_limits([-2,-1,1,5])
    # returns -5, 5
    >>> get_limits(randn(100)**2, '2 mean')
    # returns approx. 0, 2
    >>> get_limits(np.random.uniform(1,2,100), ('near', 'far'))
    # returns approx. 1, 2
    # note, that the last example is also useful if you want to do 
    # something with log(x) later (x > 0, otherwise the lower limit is 
    # calculated on the negative axis).
    """
    x = np.asanyarray(x) # be safe
    
    # check for symmetry and units #######################################
    
    ## do we have both positive and negative values?
    if not len(x): raise ValueError, "x is empty!"
    sign_min = np.sign(np.nanmin(x))
    sign_max = np.sign(np.nanmax(x))

    ## try cases with one limit
    symmetric = True
    lims      = None # only if we have two different limits
    try:
        lim  = _parse_limit(lim)
    ## ok, now cases with two limits
    except:
        lims  = [_parse_limit(l) for l in lim]
        if (
                    (lims[0] == lims[1])
                and (sign_min == sign_max)
                and (sign_min * sign_max != 0)
            ):
            # the user clearly expected to get different upper and lower
            # limits, but forgot that all points lie in the same half-plane
            # -> keep symmetric which returns zero for one limit in 
            # this case
            lim = lims[0]
            # just to be clear(ly redundant):
            lims = None
            symmetric = True
        else:
            # truly asymmetric limits
            symmetric = False
   
    # apply units ########################################################
    if symmetric:
        labs = _apply_limit_abs_unit(x, *lim)
        # don't return completely empty side...
        if sign_min < 0:
            lmin = -labs
        else:
            lmin = 0
        if sign_max > 0:
            lmax = labs
        else:
            lmax = 0
    else:
        # we have different limits
        if sign_min == sign_max:
            # still use abs because max should be the most extreme 
            lmin = sign_min * _apply_limit_abs_unit(x, *lims[0])
            lmax = sign_max * _apply_limit_abs_unit(x, *lims[1])
        elif sign_min < sign_max:
            # ok, so we have to do the slow masking for + AND/OR -
            if sign_min < 0:
                lmin = -_apply_limit_abs_unit(x[x<=0], *lims[0])
            else:
                lmin = 0
            # end if
            if sign_max > 0:
                lmax =  _apply_limit_abs_unit(x[x>=0], *lims[1])
            else:
                lmax = 0
        else:
            raise ValueError, "I don't understand the limit %s"%str(lim)
    # done
    return lmin, lmax 
  
#def date_bins(from_dt, periodicity, periods_or_to_dt=1, snap_month=True):
#    """Return edges n-year or n-month bins.
#    
#    Parameters:
#    ===========
#    from_dt: str, datetime
#        Start date
#    periodicity: str
#        E.g. '3m', '1y', ...
#    periods_or_to_dt: int, str, datetime
#        Number of periods (if int) or limit for the last date
#        to be generated. E.g. 2, '2015-01', ...
#    snap_month:
#        If a stop date was passed with periods_or_to_dt,
#        include last month if start day of month - stop day of month > 25.
#        Otherwise, all dates will have the same day as from_dt, and days 
#        are ignored.
#    """
#    import pandas as pd
#
#    # dates
#    start = pd.to_datetime(from_dt).date()
#    # running date [y, m, d]
#    d = np.array([start.year, start.month, start.day], dtype=int)
#    # date increment
#    period = parse_unit(periodicity.lower())
#    if period[-1] == 'y':
#        dp = np.array([period[0],0,0], dtype=int)      
#    elif period[-1] == 'm':
#        dp = np.array([0,period[0],0], dtype=int)
#    else:
#        raise NotImplementedError, (
#            "periodicity must be 'y' (year) or 'm' (month)!"
#        )
#    # number of periods
#    if is_number_like(periods_or_to_dt):
#        periods = int(periods_or_to_dt)
#        stop = None
#    else:
#        stop = pd.to_datetime(periods_or_to_dt).date()
#        #print start, stop, stop.day - start.day
#        # get number of months
#        months = (
#            12 * (stop.year - start.year)
#            + stop.month - start.month
#        )
#        if (
#                snap_month 
#                and ((stop.day - start.day) > 25)
#            ):
#            months += 1
#        # convert months to periods
#        if period[-1] == 'y':
#            periods = months / 12 / period[0]
#        elif period[-1] == 'm':
#            periods = months / period[0]
#
#    # store dates
#    edges = np.zeros([periods+1, 3], dtype=int)
#    # fill in dates
#    p = 0
#    for p in range(periods+1):
#        edges[p, :] = d[:]
#        d += dp
#        if d[1] > 12:
#            d[0] += 1
#            d[1] -= 12
#    # return edges as dates
#    return [pd.datetime(*e).date() for e in edges]

#def timestamps_to_date(times):
#    """Return list of dates corresponding to each timestamp.
#    Much faster than DatetimeIndex.date!!!"""
#    from pandas import DatetimeIndex
#    from datetime import date
#    di = DatetimeIndex(times)
#    return [date(*i) for i in zip(di.year, di.month, di.day)]
    
##########################################################################
# other
##########################################################################  

#def deepupdate(a, b, inplace=False, merge_lists=False):
#    '''Return new dict from a deeply updated by b'''
#    if not is_map_like(b):
#        if merge_lists and hasattr(a, '__setitem__'):
#            return a + b
#        else:
#            return b
#    if not inplace:
#        res = deepcopy(a)
#    else:
#        res = a
#    for k, v in b.iteritems():
#        if k in res:
#            if is_map_like(res[k]):
#                res[k] = deepupdate(res[k], v, merge_lists=merge_lists)
#            elif merge_lists and hasattr(res[k], '__setitem__'):
#                res[k] += v
#            else:
#                res[k] = deepcopy(v)
#        else:
#            res[k] = deepcopy(v)
#
#    if not inplace:
#        return res
    
#def pairs(elements, exclude_permutation=True, exclude_identical=True):
#    """Return all unique tuples (i,j) of elements i, j in l.
#    
#    Parameters
#    ==========
#    l: list
#        List of elements
#    exclude_permutation: bool [optional]
#        Don't return both (i,j) and (j,i). Default: True
#    exclude_identical: bool [optional]
#        Don't return (i, j=i). Default: True.
#
#    """
#    l = list(elements)
#    ix = range(len(l))
#    shift_j = int(exclude_permutation)
#    return sum([[
#        (l[i],l[j]) 
#        for j in ix[shift_j * i:]
#        if not (exclude_identical and i == j)
#    ] for i in ix], [])
#    