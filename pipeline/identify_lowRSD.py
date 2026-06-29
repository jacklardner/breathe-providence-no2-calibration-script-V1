#imports
import pandas as pd
import matplotlib.pyplot as plt

def identify_low_rsd(quant_df, concentration_cutoff=15, percentile=10):
    site_cols = ["dpw", "pema", "pha"]

    rsd_df = quant_df.copy()

    #RSD calculations
    rsd_df["mean_no2"] = rsd_df[site_cols].mean(axis=1)
    rsd_df["std_no2"] = rsd_df[site_cols].std(axis=1)
    rsd_df = rsd_df[rsd_df["mean_no2"] > 0].copy()
    rsd_df["rsd"] = (
        rsd_df["std_no2"] /
        rsd_df["mean_no2"]
    ) * 100

    overall_cutoff = rsd_df["rsd"].quantile(percentile / 100)

    original_low_rsd = rsd_df[
        rsd_df["rsd"] <= overall_cutoff
    ].copy()

    #split based on cutoff to ensure representation of lower concentrations
    low_conc = rsd_df[
        rsd_df["mean_no2"] < concentration_cutoff
    ].copy()

    high_conc = rsd_df[
        rsd_df["mean_no2"] >= concentration_cutoff
    ].copy()

    #percentile cutoffs
    low_rsd_cutoff = low_conc["rsd"].quantile(percentile / 100)
    high_rsd_cutoff = high_conc["rsd"].quantile(percentile / 100)


    low_conc_filtered = low_conc[
        low_conc["rsd"] <= low_rsd_cutoff
    ].copy()
    high_conc_filtered = high_conc[
        high_conc["rsd"] <= high_rsd_cutoff
    ].copy()

    #combine timeframes
    low_rsd_df = pd.concat(
        [low_conc_filtered, high_conc_filtered],
        ignore_index=True
    )

    low_rsd_df = low_rsd_df.sort_values(
        "datetime_utc"
    )

    #create histograms plots, one without the data split and one with the split
    site_names = ["dpw", "mjf", "pema", "pha"]
    for site in site_names:
        #no data split
        plt.figure(figsize=(6,4))

        plt.hist(original_low_rsd[site], bins=30)

        plt.xlabel("NO2 (ppb)")
        plt.ylabel("Count")
        plt.title(f"{site.upper()} - Original Low RSD Selection")

        plt.tight_layout()

        file_loc = "final PDF report/visuals for PDF/" + str(site) + "_presplit.png"
        plt.savefig(file_loc, dpi=300)

        plt.close()

        #including data split
        plt.figure(figsize=(6,4))

        plt.hist(low_rsd_df[site], bins=30)

        plt.xlabel("NO2 (ppb)")
        plt.ylabel("Count")
        plt.title(f"{site.upper()} - Concentration Split")

        plt.tight_layout()

        file_loc = "final PDF report/visuals for PDF/" + str(site) + "_postsplit.png"
        plt.savefig(file_loc, dpi=300)
        plt.close()

    summary = {
        "concentration_cutoff": concentration_cutoff,
        "percentile": percentile,
        "low_concentration_rsd_cutoff": low_rsd_cutoff,
        "high_concentration_rsd_cutoff": high_rsd_cutoff,
        "selected_hours": len(low_rsd_df),
        "total_hours": len(rsd_df),
        "selected_fraction":
            len(low_rsd_df) / len(rsd_df)
            if len(rsd_df) > 0 else 0
    }

    return low_rsd_df, rsd_df, summary