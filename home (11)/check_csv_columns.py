import pandas as pd
import argparse

def check_empty_columns(csv_path, columns_to_check=None):
    """
    Check if specified columns in a CSV file have any empty values.
    
    Args:
        csv_path (str): Path to the CSV file
        columns_to_check (list): List of column names to check. If None, checks all columns.
    
    Returns:
        dict: Dictionary with column names as keys and boolean values indicating if column has empty values
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # If no columns specified, check all columns
        if columns_to_check is None:
            columns_to_check = df.columns.tolist()
        
        # Check each specified column
        results = {}
        for column in columns_to_check:
            if column not in df.columns:
                print(f"Warning: Column '{column}' not found in CSV file")
                continue
                
            # Check for empty values (NaN, None, empty strings)
            has_empty = df[column].isna().any() or (df[column] == '').any()
            results[column] = not has_empty
            
            # Print status for each column
            status = "OK" if not has_empty else "HAS EMPTY VALUES"
            print(f"Column '{column}': {status}")
            
        return results
            
    except FileNotFoundError:
        print(f"Error: File '{csv_path}' not found")
        return {}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {}

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Check for empty values in CSV columns')
    parser.add_argument('csv_path', help='Path to the CSV file')
    parser.add_argument('--columns', nargs='+', help='List of columns to check (optional)')
    
    args = parser.parse_args()
    
    # Run the check
    check_empty_columns(args.csv_path, args.columns)

if __name__ == "__main__":
    main() 