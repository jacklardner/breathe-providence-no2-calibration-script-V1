import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from functools import reduce

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

def train_models(training_data, testing_data, quant_df, beacon_data,
                 sensor_to_site):

    predictor_cols = [
        "co_wrk_aux",
        "no2_wrk_aux",
        "no_wrk_aux",
        "o3_wrk_aux",
        "rh",
        "temp"
    ]

    site_names = {
        "Sensor01": "MJF Elementary",
        "Sensor02": "Silver Lake",
        "Sensor03": "Reservoir Ave",
        "Sensor04": "Carnevale",
        "Sensor05": "E-Cubed",
        "Sensor06": "Rochambeau",
        "Sensor07": "Smith Hill",
        "Sensor08": "Alpert Med",
        "Sensor09": "DPW",
        "Sensor10": "Zuccolo",
        "Sensor11": "West End",
        "Sensor12": "United Way",
        "Sensor13": "PHA",
        "Sensor14": "CCRI",
        "Sensor15": "Main St MA",
        "Sensor16": "South Prov",
        "Sensor17": "Blackstone",
        "Sensor18": "Brown-Fox Pt",
        "Sensor19": "Rock Spot",
        "Sensor20": "Rockefeller",
        "Sensor21": "Child & Family",
        "Sensor22": "Mt Pleasant",
        "Sensor23": "RIC",
        "Sensor24": "Providence Coll",
        "Sensor25": "PEMA"
    }

    results = []
    prediction_dfs = []

    for sensor, site in sensor_to_site.items():

        print(f"\n===== TRAINING {sensor} ({site}) =====")

        train_df = training_data[sensor].copy()
        test_df = testing_data[sensor].copy()

        train_df["datetime_utc"] = pd.to_datetime(train_df["datetime_utc"])
        test_df["datetime_utc"] = pd.to_datetime(test_df["datetime_utc"])

        target_col = site

        #ensure we handle missing values and send warning
        if len(train_df) == 0 or len(test_df) == 0:
            print(f"WARNING: {sensor} skipped (train={len(train_df)}, test={len(test_df)})")
            continue

        X_train = train_df[predictor_cols]
        y_train = train_df[target_col]

        X_test = test_df[predictor_cols]
        y_test = test_df[target_col]

        model = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        #remove negative predictions
        y_train_pred = np.clip(y_train_pred, 0, None)
        y_test_pred = np.clip(y_test_pred, 0, None)

        if sensor in ["Sensor01", "Sensor09", "Sensor13", "Sensor25"]:

            min_val = min(y_train.min(), y_test.min(), y_train_pred.min(),
                y_test_pred.min()
            )

            max_val = max(y_train.max(), y_test.max(), y_train_pred.max(),
                y_test_pred.max()
            )

            plt.figure(figsize=(6, 6))

            plt.scatter(y_train, y_train_pred, alpha=0.5)

            plt.plot([min_val, max_val], [min_val, max_val], "k--")

            plt.xlabel("Observed NO2 (ppb)")
            plt.ylabel("Predicted NO2 (ppb)")
            plt.title(f"{site_names[sensor]} Training Performance")

            plt.tight_layout()

            plt.savefig(
                f"final PDF report/visuals for PDF/{sensor}_training_scatter.png",
                dpi=300
            )

            plt.close()
            plt.figure(figsize=(6, 6))

            plt.scatter(y_test, y_test_pred, alpha=0.5)

            plt.plot([min_val, max_val], [min_val, max_val], "k--")

            plt.xlabel("Observed NO2 (ppb)")
            plt.ylabel("Predicted NO2 (ppb)")
            plt.title(f"{site_names[sensor]} Testing Performance")

            plt.tight_layout()

            plt.savefig(
                f"final PDF report/visuals for PDF/{sensor}_testing_scatter.png",
                dpi=300
            )

            plt.close()

        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        top1_feature = importance.iloc[0]["feature"]
        top1_importance = importance.iloc[0]["importance"]

        top2_feature = importance.iloc[1]["feature"]
        top2_importance = importance.iloc[1]["importance"]

        pred_df = pd.DataFrame({
            "datetime_utc": test_df["datetime_utc"],
            sensor: y_test_pred
        })

        prediction_dfs.append(pred_df)

        results.append({
            "sensor": sensor,
            "site": site,
            "site_name": site_names[sensor],

            "train_r2": train_r2,
            "train_mae": train_mae,
            "train_rmse": train_rmse,

            "test_r2": test_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,

            "top1_feature": top1_feature,
            "top1_importance": top1_importance,

            "top2_feature": top2_feature,
            "top2_importance": top2_importance,

            "n_train": len(train_df),
            "n_test": len(test_df)
        })

        print(f"{sensor} ({site}) done")

    results_df = pd.DataFrame(results).round({
        "train_r2": 3,
        "test_r2": 3,
        "train_mae": 2,
        "test_mae": 2,
        "train_rmse": 2,
        "test_rmse": 2,
        "top1_importance": 2,
        "top2_importance": 2
    })

    results_df = results_df.sort_values("test_r2", ascending=False)

    all_predictions = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on="datetime_utc",
            how="outer"
        ),
        prediction_dfs
    )

    all_predictions = (
        all_predictions
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )

    #add est/edt
    all_predictions["datetime_utc"] = pd.to_datetime(all_predictions["datetime_utc"])

    all_predictions.insert(1, "datetime_est",
        all_predictions["datetime_utc"] - pd.Timedelta(hours=5)
    )

    all_predictions.insert(2, "datetime_edt",
        all_predictions["datetime_utc"] - pd.Timedelta(hours=4)
    )

    #order by sensor number
    sensor_order = [f"Sensor{i:02d}" for i in range(1, 26)]

    sensor_cols = [
        sensor for sensor in sensor_order
        if sensor in all_predictions.columns
    ]
    all_predictions[sensor_cols] = all_predictions[sensor_cols].round(2)
    rename_dict = {
        sensor: site_names[sensor]
        for sensor in sensor_cols
    }

    all_predictions = all_predictions.rename(columns=rename_dict)

    ordered_columns = [
        "datetime_utc",
        "datetime_est",
        "datetime_edt",
    ] + [site_names[s] for s in sensor_cols]

    all_predictions = all_predictions[ordered_columns]

    #make table plot for results
    fig_height = max(4, 0.4 * len(results_df) + 1)

    fig, ax = plt.subplots(figsize=(16, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=results_df.values,
        colLabels=results_df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#DDDDDD")

    plt.tight_layout()

    plt.savefig(
        "final PDF report/visuals for PDF/model_metrics_table.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {}, results_df, all_predictions