#imports
from datetime import datetime, timedelta, timezone
from pipeline.pull_data import pull_data
from pipeline.identify_lowRSD import identify_low_rsd
from pipeline.build_datasets import build_datasets, plot_train_test_summary
from pipeline.train_models import train_models
from pipeline.networkwide_analysis import networkwide_analysis
from pipeline.reporting import generate_report

#*****USERS INPUT DATA HERE*****

#note that here we accept start and end dates in UTC.
#start date begins at 00:00:00 and end date ends at 23:00:00

#it is highly recommended to keep start date as 2025-10-28 and trim data down
#manually after. see readme

API_KEY = "YOUR_KEY_HERE"
START_DATE = "2025-10-28"
END_DATE = "2026-06-29"
print("\n=====Beginning calibration pipeline=====")

#*****ERROR CHECKING*****

start_dt = datetime.strptime(START_DATE, "%Y-%m-%d").date()
end_dt = datetime.strptime(END_DATE, "%Y-%m-%d").date()

current_utc = datetime.now(timezone.utc)
if current_utc.hour >= 23:
    latest_end_dt = current_utc.date()
else:
    latest_end_dt = (current_utc - timedelta(days=1)).date()

if start_dt > latest_end_dt:
    raise ValueError("Start date is in the future. Latest available"
    " calibration date is",str(latest_end_dt))

if end_dt > latest_end_dt:
    end_dt = latest_end_dt
    print("\nNote: End date provided exceeds current data. Rounding down to:"
    ,str(end_dt),"23:00:00")

if end_dt <= start_dt:
    raise ValueError("\nEND_DATE must occur after START_DATE")

earliest_quant_date = datetime(2025, 10, 28).date()
if start_dt < earliest_quant_date:
    raise ValueError("Start date is too early (before all Quants were online)"
                     "\nEarliest allowable date is 2025-10-28"
                     )

hours = ((end_dt - start_dt).days + 1) * 24
MIN_HOURS = 3000
if hours < MIN_HOURS:
    difference = int(MIN_HOURS - hours)
    recommend_end_dt = start_dt + timedelta(hours=MIN_HOURS)
    raise ValueError(
        f"Calibration requires at least "
        f"{MIN_HOURS:,} hours. of data\n\n"
        f"Current range provides "
        f"{int(hours):,} hours of data.\n\n"
        f"You are short by "
        f"{difference:,} hours.\n\n"
        f"Recommended END_DATE: "
        f"{recommend_end_dt.strftime('%Y-%m-%d')}\n\n"
        f"Generally, it is recommended to utilize a full year of calibrations"
        f" that include the time of interest\n"
    )

print("\nCalibration parameters accepted.\n")

#*****DOWNLOADING AND CLEANING DATA*****

print("\n=====DOWNLOADING AND CLEANING DATA=====")
quant_df, beacon_df, download_summary = pull_data(api_key=API_KEY,
                                                  start_date=start_dt,
                                                  end_date=end_dt)
print("\n=====DOWNLOADING AND CLEANING COMPLETE=====")

#*****IDENTIFYING LOW RSD TIME PERIODS*****

print("\n===== IDENTIFYING LOW RSD PERIODS =====")

low_rsd_df, rsd_df, rsd_summary = identify_low_rsd(
    quant_df,
    concentration_cutoff=15,
    percentile=10
)

print("\n===== IDENTIFIED LOW RSD PERIODS =====")

#*****BUILD TRAINING/TESTING DATASETS*****
#where each beacon site is calibrated to. Can be adjusted here by editing the 
#value of a sensor
sensor_to_site = {
    "Sensor03": "dpw",
    "Sensor08": "dpw",
    "Sensor09": "dpw",
    "Sensor14": "dpw",
    "Sensor16": "dpw",
    "Sensor18": "dpw",
    "Sensor21": "dpw",

    "Sensor01": "mjf",
    "Sensor17": "mjf",

    "Sensor05": "pema",
    "Sensor06": "pema",
    "Sensor07": "pema",
    "Sensor15": "pema",
    "Sensor20": "pema",
    "Sensor24": "pema",
    "Sensor25": "pema",

    "Sensor02": "pha",
    "Sensor04": "pha",
    "Sensor10": "pha",
    "Sensor11": "pha",
    "Sensor12": "pha",
    "Sensor13": "pha",
    "Sensor19": "pha",
    "Sensor22": "pha",
    "Sensor23": "pha",
}

print("\n===== BUILDING TRAINING/TESTING DATASETS =====")

training_data, testing_data, dataset_summary = build_datasets(
    beacon_data=beacon_df,
    quant_data=quant_df,
    low_rsd_data=low_rsd_df,
    sensor_to_site=sensor_to_site
)

#save datasets

for sensor, df in training_data.items():
    df.to_csv(f"data for analysis/training_{sensor}.csv", index=False)

for sensor, df in testing_data.items():
    df.to_csv(f"data for analysis/testing_{sensor}.csv", index=False)

dataset_summary.to_csv(
    "data for analysis/dataset_summary.csv",
    index=False
)

plot_train_test_summary(
    dataset_summary,
    output_path="final PDF report/visuals for PDF"
)

print("\n===== TRAINING/TESTING DATASETS COMPLETE =====")

#*****MODEL TRAINING*****

print("\n===== MODEL TRAINING HAS BEGUN =====")

model_results, results_df, prediction_df = train_models(
    training_data=training_data,
    testing_data=testing_data,
    quant_df=quant_df,
    beacon_data=beacon_df,
    sensor_to_site=sensor_to_site
)

results_df.to_csv("data for analysis/model_metrics_by_site.csv", index=False)

prediction_df.to_csv("final calibrated datasets/calibrated_no2_timeseries.csv",
                    index=False)

print("\n===== MODEL TRAINING COMPLETE =====")

#*****NETWORKWIDE ANALYSIS*****

print("\n===== NETWORKWIDE ANALYSIS =====")

nw_summary = networkwide_analysis(prediction_df)

print("\n===== NETWORKWIDE ANALYSIS COMPLETE =====")

#*****GENERATE PDF REPORT*****

print("\n===== GENERATING PDF REPORT =====")

generate_report()

print("\n===== PDF REPORT COMPLETE =====")