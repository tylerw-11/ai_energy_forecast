# Training Data Processing

### Import the necessary modules
import pandas as pd
import os
import numpy as np

### Configure pandas
pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 30)

### Loading CSVs and reading into a Pandas Series
def scan_csvs(path):
    dfs = []
    for i in os.scandir(path):
        if i.is_file():
            dfs.append(pd.read_csv(path + "/" + i.name))
    return dfs

### Data cleaning and filtering
def clean_data(dfs, date_key, key):
    start = '2019-01-01'
    end = '2025-01-01'
    filtered_dfs = []
    for df in dfs:
        filtered_df = df.loc[:, [date_key, key]].dropna()
        filtered_df[date_key] = pd.to_datetime(filtered_df[date_key], format='%d/%m/%Y')
        filtered_df = filtered_df.loc[(start <= filtered_df[date_key]) & (filtered_df[date_key] < end)]
        filtered_dfs.append(filtered_df)
    return filtered_dfs
    
### Provide wind data as an array
def get_wind_data():
    wind_dfs = scan_csvs("data/wind")

    wind_dfs = clean_data(wind_dfs, "Date", "Speed")

    wind_arr = []

    for df in wind_dfs:
        df = df.sort_values("Date")

        for date, speed in zip(df["Date"], df["Speed"]):
            wind_arr.append([
                date.year,
                date.month,
                speed
            ])

    return np.array(wind_arr, dtype=float)
    
### Provide energy data as an array
def get_energy_data():
    energy_dfs = scan_csvs("data/energy")
    energy_dfs = clean_data(
        energy_dfs, "Sett_Date", "Total"
    )

    energy_records = []

    for df in energy_dfs:
        df = df.copy()

        df["Year"] = df["Sett_Date"].dt.year
        df["Month"] = df["Sett_Date"].dt.month

        monthly = (
            df.groupby(["Year", "Month"], as_index=False)
              .agg(total=("Total", "sum"))
        )

        for row in monthly.itertuples(index=False):
            energy_records.append(
                (row.Year, row.Month, row.total)
            )

    return np.array(energy_records, dtype=float)

### Test
if __name__ == "__main__":
    wind = pd.DataFrame(
        get_wind_data(),
        columns=["Year", "Month", "Speed"]
    )

    energy = pd.DataFrame(
        get_energy_data(),
        columns=["Year", "Month", "Total"]
    )

# el gpt is cracked bro