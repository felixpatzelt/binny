binny
=====

Bin one-dimensional data and calculate conditional expectation values. 

The main function is ``bin_df``, which accepts numpy-arrays or pandas Dataframes 
as inputs. Bins in linear, logarithmic, or quantile space are created for the 
independent variable. The dependent variable is then aggregated in these bins.

Lots of options allow to customise the binning and aggregation, calculate 
errors, etc.

The code used to be part of a larger and quite mature python 2 package. It was
recently refactored and python 3 compatibility was added. Therefore, binny
is currently considered beta software and to be used with caution.

Examples
--------

.. code:: ipython
    
    %pylab
    from binny import bin_df
    
    x = randn(10**4)
    y = randn(10**4)
    bin_df(x, by=y, bins=10).plot(marker='x')



Dependencies
------------

    - Python >= 2.7 or >= 3.6
    - NumPy
    - Pandas    
