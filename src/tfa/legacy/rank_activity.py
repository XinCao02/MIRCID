import pandas as pd

# Read the CSV file
def rank_rows_by_column(filename):
    # Load the CSV into a DataFrame
    df = pd.read_csv(filename, index_col=0)

    # Create an empty DataFrame to store the rankings
    ranks_df = pd.DataFrame(index=df.index, columns=df.columns)
    
    # Iterate over each column to calculate the rankings
    for column in df.columns:
        ranks_df[column] = df[column].rank(ascending=False, method='min')
    
    return ranks_df

# Function to output the ranks as CSV
def save_ranks(ranks_df, output_filename):
    ranks_df.to_csv(output_filename)
    print(f"Rankings saved to {output_filename}")

# Main execution
if __name__ == "__main__":
    # Input file name (you can replace this with the path to your actual CSV file)
    input_filename = 'Z_estimated.csv'
    output_filename = 'ranked_Z_estimated.csv'

    # Rank the rows within each column
    ranks_df = rank_rows_by_column(input_filename)

    # Save the rankings to a CSV file
    save_ranks(ranks_df, output_filename)

    # Optionally, print the output for preview
    print(ranks_df.head())