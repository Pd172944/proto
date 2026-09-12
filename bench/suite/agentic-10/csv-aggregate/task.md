`sales.csv` has a header row and then one row per sale:

    region,product,units,unit_price

Write a file `report.txt` in the project root with one line per region, in
alphabetical order by region, formatted exactly as:

    <region>: <total revenue>

Revenue for a row is `units * unit_price`. Format the total with two decimal places
(so 1234.5 becomes `1234.50`). Do not use a thousands separator.
