import pandas as pd
import numpy as np
import warnings
import csv
from prophet import Prophet

warnings.filterwarnings("ignore")

# Load Data
file_path = r"C:\Users\haris\Desktop\Day6\Sales_Data_for_Analysis.tsv"  # **REPLACE WITH YOUR FILE PATH**

try:
    df = pd.read_csv(file_path, sep="\t")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

# Data Cleaning and Preprocessing
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
print(f"Columns in dataset: {df.columns}")  # Check columns to find any potential issues

# Update column renaming to reflect the correct column names
df.rename(columns={
    "PERIOD": "year",
    "QTY": "Quantity",
    "TOTAL PRICE (INR)": "Item Total",
    "CURRENCY": "Currency",
    "EX RATE": "Exchange Rate"  # Updated column name
}, inplace=True)

# Check if required columns exist
required_columns = ["year", "Currency", "Item Total", "Exchange Rate"]
for col in required_columns:
    if col not in df.columns:
        print(f"Error: Column '{col}' not found in the dataset!")
        exit()

# Convert 'year' to datetime, handling errors
df["year"] = pd.to_datetime(df["year"], errors="coerce", dayfirst=True).dt.year
df.dropna(subset=["year"], inplace=True)  # Drop rows where year conversion failed
df["year"] = df["year"].astype(int)

# Standardize 'Currency' column
df["Currency"] = df["Currency"].str.strip().str.upper()

# Handle Exchange Rate and convert USD to INR if needed
if "Exchange Rate" in df.columns:
    # Convert USD to INR before processing
    df.loc[df["Currency"] == "USD", "Item Total"] *= df["Exchange Rate"]
    df["Currency"] = "INR"  # Convert all to INR
else:
    print("Warning: 'Exchange Rate' column not found, skipping currency conversion.")
    # If Exchange Rate is missing, define a default exchange rate (e.g., 75 for USD to INR conversion)
    default_exchange_rate = 75  # Example rate
    df.loc[df["Currency"] == "USD", "Item Total"] *= default_exchange_rate
    df["Currency"] = "INR"  # Convert all to INR

# Ensure data is not empty
if df.empty:
    print("No valid data available. Exiting.")
    exit()

latest_year = int(df["year"].max())  # Get latest year for forecasting

# Group Data by PART NO and year, summing Quantity and Item Total
grouped = df.groupby(["PART NO", "year"])[["Quantity", "Item Total"]].sum().reset_index()

predictions = []

# Iterate over unique part numbers to forecast
for part_no in grouped["PART NO"].unique():
    part_data = grouped[grouped["PART NO"] == part_no]

    # Skip if not enough data points for Prophet model (minimum 3)
    if len(part_data) < 3:
        print(f"Not enough data points for {part_no} to use Prophet. Skipping.")
        predictions.append([part_no, latest_year + 1, np.nan, np.nan, "INR"])
        continue

    # Prepare data for Prophet (rename columns to fit the model)
    part_data_prophet = part_data.rename(columns={"year": "ds", "Quantity": "y_quantity", "Item Total": "y_total"})

    # Remove duplicate years (ds)
    part_data_prophet = part_data_prophet.drop_duplicates(subset='ds', keep='first')

    # Convert 'ds' column to datetime
    part_data_prophet['ds'] = pd.to_datetime(part_data_prophet['ds'], format='%Y', errors='raise')

    # Quantity Forecasting
    model_quantity = Prophet()
    try:
        model_quantity.fit(part_data_prophet[['ds', 'y_quantity']].rename(columns={'y_quantity': 'y'}))
        future_quantity = pd.DataFrame({'ds': [pd.to_datetime(latest_year + 1, format='%Y')]}).astype('datetime64[ns]')
        forecast_quantity = model_quantity.predict(future_quantity)
        pred_quantity = forecast_quantity['yhat'].values[0]

        # Ensure non-negative and integer values
        pred_quantity = int(max(0, round(pred_quantity)))
    except Exception as e:
        print(f"Error with Prophet (Quantity) for {part_no}: {e}")
        pred_quantity = np.nan

    # Total Forecasting
    model_total = Prophet()
    try:
        model_total.fit(part_data_prophet[['ds', 'y_total']].rename(columns={'y_total': 'y'}))
        future_total = pd.DataFrame({'ds': [pd.to_datetime(latest_year + 1, format='%Y')]}).astype('datetime64[ns]')
        forecast_total = model_total.predict(future_total)
        pred_total = forecast_total['yhat'].values[0]

        # Ensure non-negative and integer values
        pred_total = int(max(0, round(pred_total)))
    except Exception as e:
        print(f"Error with Prophet (Total) for {part_no}: {e}")
        pred_total = np.nan

    # Append predictions to list, making sure the predicted values are integers
    predictions.append([
        part_no, 
        latest_year + 1, 
        int(pred_quantity) if not np.isnan(pred_quantity) else 0,  # Handle NaN values
        int(pred_total) if not np.isnan(pred_total) else 0,       # Handle NaN values
        "INR"
    ])

# Convert Predictions to DataFrame
predictions_df = pd.DataFrame(predictions, columns=["PART NO", "year", "Predicted Quantity", "Predicted Item Total", "Currency"])

# Handle NaN and Inf values before converting to integer
predictions_df["Predicted Quantity"].fillna(0, inplace=True)  # Replace NaN with 0
predictions_df["Predicted Item Total"].fillna(0, inplace=True)  # Replace NaN with 0

# Convert columns to integer to remove decimal points (".0")
predictions_df["Predicted Quantity"] = predictions_df["Predicted Quantity"].astype(int)
predictions_df["Predicted Item Total"] = predictions_df["Predicted Item Total"].astype(int)

# Save Predictions to TSV
output_file = r"C:\Users\haris\Desktop\Day7\predictions.tsv"  # **REPLACE WITH YOUR OUTPUT FILE PATH**
predictions_df.to_csv(output_file, index=False, sep="\t", quoting=csv.QUOTE_NONNUMERIC)

print(f"Predictions saved in TSV format at: {output_file}")
print(predictions_df)
