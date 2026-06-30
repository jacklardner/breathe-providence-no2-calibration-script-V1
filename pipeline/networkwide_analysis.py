import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import geopandas as gpd
import contextily as ctx

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "final PDF report/visuals for PDF"
EXCLUDE = ["MJF Elementary", "Blackstone"]
SITE_COORDS = {
    "Silver Lake": (41.810533, -71.447651),
    "Reservoir Ave": (41.79157, -71.42788),
    "Carnevale": (41.81678, -71.46458),
    "E-Cubed": (41.85417, -71.43444),
    "Rochambeau": (41.84621, -71.39663),
    "Smith Hill": (41.83549, -71.42217),
    "Alpert Med": (41.81899, -71.40828),
    "DPW": (41.79521, -71.39783),
    "Zuccolo": (41.82287, -71.43067),
    "WECC": (41.80439, -71.42825),
    "United Way": (41.81803, -71.44121),
    "PHA": (41.81715, -71.455331),
    "CCRI": (41.80745, -71.41341),
    "Main St MA": (41.85521, -71.40042),
    "South Prov": (41.80205, -71.41378),
    "Brown-Fox Pt": (41.82469, -71.39755),
    "Rock Spot": (41.81478, -71.42251),
    "Rockefeller": (41.82551, -71.40528),
    "Child & Family": (41.7909, -71.4031),
    "Mt Pleasant": (41.83168, -71.444401),
    "RIC": (41.84212, -71.46526),
    "Providence College": (41.84272, -71.43888),
    "PEMA": (41.85224, -71.41983),
}

def networkwide_analysis(prediction_df):
    df = prediction_df.copy()

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df["datetime_est"] = pd.to_datetime(df["datetime_est"])
    df["datetime_edt"] = pd.to_datetime(df["datetime_edt"])

    site_cols = list(df.columns[3:])
    subset = [c for c in site_cols if c not in EXCLUDE]

    #summary stats
    summary = pd.DataFrame({
        "Mean": df[site_cols].mean(),
        "Median": df[site_cols].median(),
        "Std": df[site_cols].std(),
        "Min": df[site_cols].min(),
        "Max": df[site_cols].max(),
        "5th %": df[site_cols].quantile(0.05),
        "95th %": df[site_cols].quantile(0.95),
        "Missing": df[site_cols].isna().sum(),
        "N": df[site_cols].count(),
    }).round(2)

    fig, ax = plt.subplots(figsize=(16, max(6, len(site_cols) * 0.45 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary.values,
        rowLabels=summary.index,
        colLabels=summary.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f5f6fa" if row % 2 == 0 else "white")

    plt.title("Network-Wide NO2 Summary Statistics", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/summary_stats_table.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("✓ Summary stats table saved.")

    #bixplots for all sites
    fig, ax = plt.subplots(figsize=(14, 6))
    df[site_cols].boxplot(ax=ax, rot=90, showfliers=False)
    ax.set_ylabel("NO2 (ppb)")
    ax.set_title("NO2 Distribution by Site (All Sites)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/boxplots_all_sites.png", dpi=300)
    plt.close()
    print("Boxplots (all sites) saved.")

    #boxplots subset
    fig, ax = plt.subplots(figsize=(14, 6))
    df[subset].boxplot(ax=ax, rot=90, showfliers=False)
    ax.set_ylabel("NO2 (ppb)")
    ax.set_title("NO2 Distribution by Site (Excl. MJF & Blackstone)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/boxplots_subset.png", dpi=300)
    plt.close()
    print("Boxplots (subset) saved.")

    #median NO2 all sites
    median_all = df[site_cols].median().sort_values()
    fig, ax = plt.subplots(figsize=(12, 5))
    median_all.plot.bar(ax=ax, color="steelblue", edgecolor="white")
    ax.set_ylabel("Median NO2 (ppb)")
    ax.set_title("Median NO2 by Site (All Sites)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/median_no2_bar_all.png", dpi=300)
    plt.close()
    print("Median NO2 bar (all sites) saved.")

    #median NO2 subset
    median_sub = df[subset].median().sort_values()
    fig, ax = plt.subplots(figsize=(12, 5))
    median_sub.plot.bar(ax=ax, color="steelblue", edgecolor="white")
    ax.set_ylabel("Median NO2 (ppb)")
    ax.set_title("Median NO2 by Site (Excl. MJF & Blackstone)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/median_no2_bar_subset.png", dpi=300)
    plt.close()
    print("Median NO2 bar (subset) saved.")

    #histograms
    n_sites = len(site_cols)
    ncols, nrows = 5, 5
    all_vals = df[site_cols].values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    bin_edges = np.histogram_bin_edges(all_vals, bins=20)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 14), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, col in enumerate(site_cols):
        ax = axes_flat[i]
        vals = df[col].dropna()
        ax.hist(vals, bins=bin_edges, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.set_title(col, fontsize=7, pad=2)
        ax.tick_params(labelsize=6)

    for j in range(n_sites, nrows * ncols):
        axes_flat[j].set_visible(False)

    fig.supxlabel("NO2 (ppb)", fontsize=10)
    fig.supylabel("Count", fontsize=10)
    fig.suptitle("NO2 Concentration Distributions by Site", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/histograms_5x5_grid.png", dpi=200)
    plt.close()
    print("5×5 histogram grid saved.")

    #diurnal cycle heatmap
    df["hour"] = df["datetime_est"].dt.hour
    diurnal = df.groupby("hour")[site_cols].mean()

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(
        diurnal.T.values,
        aspect="auto",
        cmap="YlOrRd",
        origin="lower",
    )
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=8)
    ax.set_yticks(range(len(site_cols)))
    ax.set_yticklabels(site_cols, fontsize=8)
    ax.set_xlabel("Hour of Day (EST)")
    ax.set_title("Mean NO2 by Hour of Day & Site (Diurnal Heatmap)")
    plt.colorbar(im, ax=ax, label="Mean NO2 (ppb)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/diurnal_heatmap.png", dpi=300)
    plt.close()
    print("Diurnal heatmap saved.")

    #monthly trend heatmap
    df["month"] = df["datetime_est"].dt.to_period("M")
    monthly = df.groupby("month")[site_cols].mean()
    month_labels = [str(m) for m in monthly.index]

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(
        monthly.T.values,
        aspect="auto",
        cmap="YlOrRd",
        origin="lower",
    )
    ax.set_xticks(range(len(month_labels)))
    ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(site_cols)))
    ax.set_yticklabels(site_cols, fontsize=8)
    ax.set_xlabel("Month")
    ax.set_title("Mean NO2 by Month & Site (Monthly Trend Heatmap)")
    plt.colorbar(im, ax=ax, label="Mean NO2 (ppb)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_trend_heatmap.png", dpi=300)
    plt.close()
    print("Monthly trend heatmap saved.")

    #correlation heatmap
    corr = df[site_cols].corr()
    #set any negative correlations to 0
    corr_clipped = corr.clip(lower=0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_clipped.values, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(site_cols)))
    ax.set_yticks(range(len(site_cols)))
    ax.set_xticklabels(site_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(site_cols, fontsize=7)
    ax.set_title("Site-to-Site NO2 Correlation", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Pearson r (clipped at 0)")

    #annotation for cells
    for i in range(len(site_cols)):
        for j in range(len(site_cols)):
            val = corr_clipped.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5,
                    color="black" if val < 0.7 else "white")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=200)
    plt.close()
    print("Correlation heatmap saved.")

    #95th percentile map
    pct95 = df[site_cols].quantile(0.95).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    pct95.plot.bar(ax=ax, color="tomato", edgecolor="white")
    ax.set_ylabel("NO2 (ppb) — 95th Percentile")
    ax.set_title("95th Percentile NO2 Exposure by Site")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pct95_exposure_bar.png", dpi=300)
    plt.close()
    print("95th percentile bar chart saved.")

    #hierarchal clustering
    corr_sub = df[subset].corr()
    dist_matrix = 1 - corr_sub.values
    np.fill_diagonal(dist_matrix, 0)
    dist_condensed = squareform(np.clip(dist_matrix, 0, None))

    Z = linkage(dist_condensed, method="ward")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, labels=subset, leaf_rotation=90, leaf_font_size=9, ax=ax)
    ax.set_title("Hierarchical Clustering of NO2 Sites\n(Ward linkage, 1 − Pearson r distance, excl. MJF & Blackstone)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hierarchical_clustering.png", dpi=300)
    plt.close()
    print("Hierarchical clustering dendrogram saved.")



    #standardized map values
    map_sites = [s for s in subset if s in SITE_COORDS]
    if map_sites:
        lats = [SITE_COORDS[s][0] for s in map_sites]
        lons = [SITE_COORDS[s][1] for s in map_sites]
        medians = np.array([df[s].median() for s in map_sites])

        #use z score standardization
        network_mean = medians.mean()
        network_std  = medians.std()
        z_scores = (medians - network_mean) / network_std

        gdf = gpd.GeoDataFrame(
            {"site": map_sites, "z_median_no2": z_scores},
            geometry=gpd.points_from_xy(lons, lats),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        max_abs = np.abs(z_scores).max()

        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(
            ax=ax,
            column="z_median_no2",
            cmap="RdBu_r",
            vmin=-max_abs,
            vmax=max_abs,
            markersize=120,
            legend=True,
            legend_kwds={"label": "Median NO2 (z-score, network mean = 0)", "shrink": 0.6},
            zorder=5,
            edgecolor="black",
            linewidth=0.5,
        )

        for _, row in gdf.iterrows():
            ax.annotate(
                row["site"],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                zorder=6,
            )

        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=13)
        except Exception as e:
            print(f"  (Basemap unavailable: {e})")

        ax.set_axis_off()
        ax.set_title(
            "NO2 Monitoring Sites — Median Concentration (Z-Score)\n"
            "Standardized to Subset Network Mean (Excl. MJF & Blackstone)",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/site_map.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Site map saved.")
    else:
        print("\nSkipped map: no matching coordinates found in SITE_COORDS.")

    print("\nAll visuals saved to:", OUTPUT_DIR)
    return summary