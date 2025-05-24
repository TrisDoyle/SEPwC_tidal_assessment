# SEPwC Tidal Analysis Coursework

## Introduction

Your task is to write code to read in multiple tidal data files (each a different year) 
for a single tidal station. You should then calculate the M2 and S2 tidal components
and also the rate of sea-level rise per year, for that station.

## The tests

The test suite uses two data files from the 1940's from the
Aberdeen tide gauge station. The test will check reading in these
data, joining multiple years and calculating tidal constituents.

You can run the tests by running `pytest` or `pytest test/test_tides.py`
from the main directory. Try it now, before you make any changes!

You can run a single test using:

```bash
pytest test/test_tides.py::TestTidalAnalysis::test_reading_data
```

You can run the unit tests only:

```bash
pytest test/test_tides.py::TestTidalAnalysis
```

The regression tests check the whole code:

```bash
pytest test/test_tides.py::TestRegression
```

## The data

There are three directories with data for Whitby, Aberdeen and Dover for the years
2000 to 2019. Your code must be able to read in all the data files in one of those
directories and perform the analysis by the user suppying the folder. You should be able to run:

```bash
python3 tidal_analysis.py data/whitby
```

for example to calculate the stats for Whitby. The program should print the tidal data, the sea-level rise and
the longest contiguous period of data (i.e. without any missing data) from 
the data loaded. 

Each file in the folder contains the tidal elevation and residual for a year.
There are records every 15 to 60 minutes. The files are laid out like this:

```bash
Port:              P038
Site:              Aberdeen
Latitude:          57.14325
Longitude:         -2.07451
Start Date:        01JAN1946-00.00.00
End Date:          31DEC1946-23.00.00
Contributor:       National Oceanography Centre, Liverpool
Datum information: The data refer to Admiralty Chart Datum (ACD)
Parameter code:    ASLVZZ01 = Surface elevation (unspecified datum) of the water body                      
  Cycle    Date      Time      ASLVZZ01     Residual  
 Number yyyy mm dd hh mi ssf           f            f 
     1) 1946/01/01 00:00:00      3.6329      -0.1522  
     2) 1946/01/01 01:00:00      3.4195      -0.1547  
     3) 1946/01/01 02:00:00      2.9013      -0.2043  
     4) 1946/01/01 03:00:00      2.2612      -0.2170  
     5) 1946/01/01 04:00:00      1.6821      -0.2059  
     6) 1946/01/01 05:00:00      1.2859      -0.1926  
```
The headers should be ignored by your program. The data should consist of the at minimum
a `datetime` column and a column called `Sea Level` which is the `ASLVZZ01` data.

The UK Tidal Database has the following M2 and S2 amplitudes for our three stations
of interest:

| Station Name    |   M2 Amplitude   | S2 Amplitude  |
|-----------------|------------------|---------------|
| Whitby          | 1.659 m          | 0.558 m       |
| Aberdeen        | 1.307 m          | 0.441 m       |
| Dover           | 2.243 m          | 0.701 m       |

Your code should return value similar to these, but they don't have to 
be identical.

## Hints and tips

Use `pandas` to load data and manage most of the data wrangling. 

The `scipy.stats' module can do the linear regression to work out sea-level rise. You may find
it easier to work out the rise per day and multiply by 365 to get metres per year. 

`uptide` can calculate the tidal constuents, but any `nan` (i.e missing data) must be 
removed prior to working out tidal constiuents. 

To search for files of a certain type within a directory, use the ``glob`` module. 

Regular experssions might be useful for removing the dodgy values (`N`, `M`, `T`), so
something like:

```python
my_data.replace(to_replace=".*M$",value={'A':np.nan},regex=True,inplace=True)
```
will replace any value ending in M in the column `A` in the pandas dataframe `my_data`. 
Note I've done this `inplace` so I don't need to reassign into another variable.

To convert datetimes to numbers to do a linear regression (see above), there are a few ways, 
but using the [`date2num`](https://matplotlib.org/stable/api/dates_api.html#matplotlib.dates.date2num)
function in `matplotlib.dates`, which converts a datetime
to the number of days since 01-01-1970. 


## The rules

You cannot alter any of the assert comments in `test/test_tides.py`

If you alter any function names in the main code, you *can* alter the name
in the test file to match; however the rest of the test must remain unchanged. 
This will be checked.

If you wish to add more tests, please do, but place them in a separate file
in the `test` directory. Remember to name the file `test_something.py`. You must
also make sure the `class` name(s) are different to those in `test/test_tides.py`.

You can also add extra functionality, but the command-line interface must pass
the tests set.
