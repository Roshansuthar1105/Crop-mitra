import pandas as pd

def check_csv_ranges(csv_file):
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Select only numerical columns
        numeric_cols = df.select_dtypes(include=['number'])

        # Check min and max for each column
        ranges = numeric_cols.agg(['min', 'max'])

        print("Column-wise Min & Max Values:")
        print(ranges)

    except Exception as e:
        print(f"Error: {e}")

# Example Usage
csv_file = "plant(IBM - Z).csv"  # 🔄 Change this to your actual file path
check_csv_ranges(csv_file)
