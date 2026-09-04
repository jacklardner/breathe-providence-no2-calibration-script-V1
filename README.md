# breathe-providence-no2-calibration-script

See "calibration summary v1.pdf" for a highly detailed description of the
calibration process.

This application is a streamlined script that pulls calibrtated hourly NO2 data
from the Breathe Providence low cost air sensor network. Users can change the
parameters in the main.py file to download calibrated NO2 data as needed.

Breathe Providence maintains 25 low cost BEACO2N air sensors throughout the
city. Providence as a city lacks a dense network of government run reference
grade air sensors that are utilized in remote calibration techniques. Only a
single DEM run reference site capable of measuring NO2 is present outside (and
"upwind") of Providence. To circumvent this, we utilize a network of QuantAQ
MODULAIR sensors that are equipped with labratory based backend calibrations.
These four Modulairs are collocated with BEACO2N sensors throughout the city and
provide the reference time series of NO2 that the sensors are calibrated
against.

The file can be run by entering the start date, end date, and QuantAQ API key in
the main.py file and running the main.py file. If user is outside the Breathe
Providence QuantAQ network, please contact Meredith Hastings
(meredith_hastings@brown.edu) or Jack Lardner (john_lardner@brown.edu) for more
details on how to get an API key and/or data.

At least 3000 hours of data are required to be pulled as 10% of the hourly data
is used to train the gradient boosting model that calibrates the rest of the
time series. It is generally recommended to pull as much data as possible,
using a starting date of 10-28-2025 and pulling as close to the present as
possible. This allows for the maximum amount of training data for the models.
Data cannot be calibrated before 10-28-2025 due to the four QuantAQ Modulairs
not being online continuously before that date. In future iterations of the
script, we hope to calibrate past BEACO2N data using models trained on a large
amount of relatively present data.

The terminal will print updates on the calibration process. A final dataset is
produced in the "final calibrated datasets" folder, while a PDF report is
generated in the "final PDF report" folder. Additional data is generated in the
"data for analysis folder", including training/testing data for each site.

This remote calibration technique is based on work by Winters et al. (\*). Our
approach is adjusted (as listed previously) to utilize a network of QuantAQ
MODULAIR air sensors in the inner city to identify low relative standard
deviation (RSD) time periods, and to calibrate each BEACO2N sensor to its
nearest MODULAIR. The RIDEM reference site located in East Providence
experiences a much lower distribution of NO2 data and thus fails as a reference
site to calibrate inner-city BEACO2N sensors against. The updated MODULAIR NO2
calibrations are deemed trustworthy based on performance of our MODULAIR that
is collocated with the DEM site in East Providence.

(\*) (Winter, Anna R., et al. “A Scalable Calibration Method for Enhanced
Accuracy in Dense Air Quality Monitoring Networks.” Environmental Science &
Technology, vol. 59, no. 5, 28 Jan. 2025, pp. 2599–2610,
doi:10.1021/acs.est.4c08855. )