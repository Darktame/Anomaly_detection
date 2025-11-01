Soil Sensor Anomaly Detection using LSTM Autoencoder
This project uses a Long Short-Term Memory (LSTM) Autoencoder to identify anomalies in time-series data from soil sensors. The model is built with TensorFlow/Keras and trained to learn the "normal" behavior of the sensors. It then flags any data points that deviate significantly from this learned pattern as anomalies.

This notebook was prepared as part of a university project to analyze IoT sensor data for potential sensor faults or environmental irregularities.

Methodology
The notebook follows a clear workflow for time-series anomaly detection:

Data Preprocessing:

Loads the soil_data_incl_rain_v3.csv dataset.

Cleans the data by converting timestamps, sorting, and resampling to a consistent 1-hour (1H) frequency.

Missing values (e.g., for hours with no readings) are filled using linear interpolation.

All sensor features (Humidity, Atmospheric_Temp, etc.) are standardized using StandardScaler.

Sequence Generation:

The time-series data is converted into overlapping sequences (or "windows") of 30 timesteps each. This prepares the data for the LSTM model.

The data is then split into a training set (80%) and a test set (20%).

Model Architecture:

An LSTM Autoencoder is built.

Encoder: An LSTM(64) layer compresses the 30-timestep sequence into a single summary vector.

Bridge: A RepeatVector layer repeats this summary vector 30 times.

Decoder: Two LSTM layers are stacked to decompress the summary sequence and reconstruct the original 30-timestep input.

Training and Detection:

The model is trained on the X_train data, learning to reconstruct its own input by minimizing the Mean Squared Error (MSE).

The trained model is then used to predict (reconstruct) the X_test data.

A statistical anomaly threshold is calculated (mean error + 3 standard deviations).

Results and Analysis:

Any sequence from the test set with a reconstruction error (MSE) higher than the threshold is flagged as an anomaly.

The results are analyzed by showing which features (e.g., Soil_Moisture, Soil_Temp) contributed the most to the error for each detected anomaly.

The notebook includes several visualizations (histograms, time-series plots, and heatmaps) to display the results.

🚀 Getting Started
Prerequisites
You will need the following Python libraries. You can install them all using pip.

pandas

numpy

scikit-learn

tensorflow

matplotlib

seaborn

Bash

pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
Usage
Clone or download this repository.

Please make sure you have the dataset soil_data_incl_rain_v3.csv in the right path (the notebook assumes /content/soil_data_incl_rain_v3.csv, which is common in Google Colab).

Open the Soil_Tech_Project.ipynb notebook in Jupyter or Google Colab.

Run the cells sequentially from top to bottom.
