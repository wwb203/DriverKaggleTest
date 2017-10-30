# DriverKaggleTest

```
git clone git@github.com:weihuacern/DriverKaggleTest.git
cd DriverKaggleTest
mkdir data
```
csv data is too big, download to local my yourself
copy csv to data directory

then
```python
python Kernel.py
```

First aim is to find the best way to deal with missing data. Here i use a random forest method to score the missing data refill scheme
The baseline value (nothing replaced, remain to be -1) of the error is 0.0823917474967
