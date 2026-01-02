import argparse
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze prediction results.")
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='Path to the CSV file containing predictions.')
    parser.add_argument('--ref_logfc_column', type=str, required=True,
                        help='Name of the column containing reference log fold change values.')
    parser.add_argument('--pred_logfc_column', type=str, required=True,
                        help='Name of the column containing predicted log fold change values.')
    parser.add_argument('--separator', type=str, default=',',
                        help='Separator used in the CSV file (default: ",").')
    return parser.parse_args()


def analyze_predictions(predictions_file, ref_logfc_column, pred_logfc_column, separator):
    # Load the predictions data
    if separator == 'tab':
        separator = '\t'
    df = pd.read_csv(predictions_file, sep=separator)

    # Ensure the required columns are present
    if ref_logfc_column not in df.columns or pred_logfc_column not in df.columns:
        raise ValueError("Specified columns not found in the predictions file.")

    # Calculate correlation between reference and predicted log fold changes
    correlation = df[ref_logfc_column].corr(df[pred_logfc_column])

    # Print analysis results
    print(f"Correlation between {ref_logfc_column} and {pred_logfc_column}: {correlation:.4f}")


if __name__ == "__main__":
    args = parse_arguments()
    analyze_predictions(args.predictions_file, args.ref_logfc_column, args.pred_logfc_column, args.separator)