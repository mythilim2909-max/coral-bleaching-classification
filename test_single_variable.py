import requests
import pandas as pd
from io import StringIO

# Testing ONE single variable, on the actual source server, for ONE known real site
url = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.csv?CRW_DHW[(2019-08-03T00:00:00Z):1:(2019-10-26T00:00:00Z)][(20.88104)][(-156.69102)]"

print("Testing URL:")
print(url)
print()

response = requests.get(url, timeout=30)
print("Status code:", response.status_code)
print()
print("Response text (first 1000 characters):")
print(response.text[:1000])