binny
=====

Bin one-dimensional data and calculate conditional expectation values. 

The main function is ``bin_df``, which accepts numpy-arrays or pandas Dataframes 
as inputs. Bins in linear, logarithmic, or quantile space are created for the 
independent variable. The dependent variable is then aggregated in these bins.

Lots of options allow to customise the binning and aggregation, calculate 
errors, etc.


Examples
--------

.. code:: ipython
    
    %pylab
    from binny import bin_df
    
    x = randn(10**4)
    y = x**2 + randn(10**4)
    bin_df(y, by=x, bins=11).plot(marker='x')
    
    # now plot the same data using bins containing an equal number of events
    # (quantile bins)
    bin_df(y, by=x, bins=11, space='q').plot(marker='+')


Installation
------------

	pip install binny


Dependencies
------------

    - Python >= 2.7 or >= 3.6
    - NumPy
    - Pandas    
