@echo off


set OUTPUT_DIRPATH=scrape
set CSV_FILEPATH=book30-listing-test.csv

python bookcoverdown.py %OUTPUT_DIRPATH% %CSV_FILEPATH%

pause