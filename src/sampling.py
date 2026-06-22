import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_sample(
    input_file,
    output_file,
    sample_size=12000
):
    """
    Creates a stratified sample while preserving
    the original product distribution.
    """

    try:

        df = pd.read_csv(input_file)

        if len(df) < sample_size:
            sample_size = len(df)

        sample_df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df["Product"],
            random_state=42
        )

        sample_df.to_csv(
            output_file,
            index=False
        )

        print(
            f"Saved sample dataset to {output_file}"
        )

        return sample_df

    except FileNotFoundError:
        print(
            f"Error: {input_file} not found."
        )