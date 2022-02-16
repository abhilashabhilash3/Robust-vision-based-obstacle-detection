#####################################################################################
# Run this instead of 04x_rasterization_and_models.py to utilize multiple CPU cores #
#####################################################################################

TOTAL=2 # Number of threads
for ((i=0;i<=TOTAL-1;i++)); do
    gnome-terminal -e "/home/abhilash/Documents/anomaly_detector/.env/bin/python /home/abhilash/Documents/anomaly_detector_0.7_1.0/anomaly_detector/anomaly_detector/scripts/04x_rasterization_and_models.py --index $i --total $TOTAL"
done