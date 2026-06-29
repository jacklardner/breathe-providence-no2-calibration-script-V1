import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

def build_datasets(beacon_data, quant_data, low_rsd_data, sensor_to_site):

    training_dict = {}
    testing_dict = {}

    summary_rows = []
    low_rsd_set = set(low_rsd_data["datetime_utc"])

    for sensor_id, beacon_df in beacon_data.items():

        site = sensor_to_site[sensor_id]

        print(f"\n--- Processing {sensor_id} -> {site} ---")

        #merge the beacon and the quants
        df = beacon_df.merge(
            quant_data[["datetime_utc", site]],
            on="datetime_utc",
            how="inner"
        )

        before = len(df)
        df = df.dropna()
        after = len(df)

        dropped = before - after

        test_df = df[~df["datetime_utc"].isin(low_rsd_set)].copy()
        train_df = df[df["datetime_utc"].isin(low_rsd_set)].copy()

        training_dict[sensor_id] = train_df
        testing_dict[sensor_id] = test_df

        summary_rows.append({
            "sensor": sensor_id,
            "site": site,
            "total": len(df),
            "train": len(train_df),
            "test": len(test_df),
            "dropped_na_rows": dropped,
            "test_fraction": len(test_df) / len(df) if len(df) > 0 else 0
        })

        print(
            f"{sensor_id}: train={len(train_df):,}, "
            f"test={len(test_df):,}, dropped={dropped:,}"
        )

    summary_df = pd.DataFrame(summary_rows)

    return training_dict, testing_dict, summary_df

def plot_train_test_summary(summary_df, output_path):
    plt.figure(figsize=(10, 6))

    x = range(len(summary_df))

    plt.bar(x, summary_df["train"], label="Train")
    plt.bar(
        x,
        summary_df["test"],
        bottom=summary_df["train"],
        label="Test"
    )

    plt.xticks(
        x,
        summary_df["sensor"],
        rotation=45,
        ha="right"
    )

    plt.ylabel("Hours")
    plt.title("Train/Test Split by Sensor (Low RSD Holdout)")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{output_path}/train_test_counts.png",
        dpi=300
    )

    plt.close()