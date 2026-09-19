import random
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from sklearn.model_selection import train_test_split

from project_utils import utils as util
from project_utils import parameters as param
from project_utils import model_utils as mu

# Reproducibility
np.random.seed(101)
random.seed(201)
tf.random.set_seed(333)

# Restrict TensorFlow to a single CPU thread for reproducible runs
session_conf = tf.compat.v1.ConfigProto(device_count={'CPU': 1})
sess = tf.compat.v1.Session(config=session_conf)

# Paths and constants
DATA_DIR = Path("./commondata")
R1_DATA_DIR = Path("./data")  # outputs of 0-process_data.ipynb
OUTPUT_DIR = Path("./processed_data")

HGT_850_FILE = DATA_DIR / "hgt_anomalies_850-jjas-22.nc"
SLP_FILE = DATA_DIR / "slp_anomalies-jjas-22.nc"
UWND_FILE = R1_DATA_DIR / "uwnd_anomalies_r1_1979-2022.nc"
VWND_FILE = R1_DATA_DIR / "vwnd_anomalies_r1_1979-2022.nc"
SHUM_FILE = R1_DATA_DIR / "shum_anomalies_r1_1979-2022.nc"

REGIONS = ['IP', 'WCI', 'NWI', 'CNE', 'NEI', 'HR']  # IMD homogeneous rainfall regions

def load_u_v_slp_input():
    """
    Build the CNN predictor array from u-wind, v-wind, and SLP anomalies.

    Returns
    -------
    numpy.ndarray
        Array of shape [time, lat, lon, 3], channel order (u, v, slp).
    """
    u_data = np.array(xr.open_dataset(UWND_FILE)["uwnd_anom"]).reshape(
        param.n, param.nlats, param.nlons, 1
    )
    v_data = np.array(xr.open_dataset(VWND_FILE)["vwnd_anom"]).reshape(
        param.n, param.nlats, param.nlons, 1
    )
    slp_data = np.array(xr.open_dataset(SLP_FILE)["slp_anom"]).reshape(
        param.n, param.nlats, param.nlons, 1
    )
    return np.concatenate([u_data, v_data, slp_data], axis=3)


def load_hgt_shum_slp_input():
    """
    Build the CNN predictor array from 850 hPa geopotential height,
    specific humidity, and SLP anomalies.

    Returns
    -------
    numpy.ndarray
        Array of shape [time, lat, lon, 3], channel order (hgt850, shum, slp).
    """
    hgt_data = np.array(xr.open_dataset(HGT_850_FILE)["hgt_anom"]).reshape(
        param.n, param.nlats, param.nlons, 1
    )
    shum_data = np.array(xr.open_dataset(SHUM_FILE)["shum_anom"]).reshape(
        param.n, param.nlats, param.nlons, 1
    )
    slp_data = np.array(xr.open_dataset(SLP_FILE)["slp_anom"]).reshape(
        param.n, param.nlats, param.nlons, 1
    )
    return np.concatenate([hgt_data, shum_data, slp_data], axis=3)

def train_region(region, x_dat, subdir, output_dir=OUTPUT_DIR, n_channels=3):
    """
    Train and evaluate the 3-channel CNN for a single IMD homogeneous
    rainfall region.

    Parameters
    ----------
    region : str
        Region abbreviation, e.g. "IP".
    x_dat : numpy.ndarray
        Predictor array from `load_u_v_slp_input` or `load_hgt_shum_slp_input`.
    subdir : str
        Output subdirectory for this variable combination
        (e.g. "U-V-SLP", "Shum-SLP-HGT").
    output_dir : pathlib.Path
        Root directory for processed data and results.
    n_channels : int
        Number of input channels for the CNN.

    Returns
    -------
    (keras.callbacks.History, pandas.DataFrame)
        Training history and per-sample predictions.
    """
    region_path = output_dir / region
    subdir_path = region_path / subdir
    subdir_path.mkdir(parents=True, exist_ok=True)

    initial_weights_path = output_dir / "initial_weights.h5"
    if not initial_weights_path.exists():
        print("Initial weights not found. Creating dummy weights...")
        mu.build_model(input_channels=n_channels).save_weights(initial_weights_path)

    y_dat = util.get_precip_classes(
        pd.read_csv(region_path / f"region_mean_precip_{region}.csv")["rain"],
        q=[0.95],
    )
    y_dat_onehot = util.onehot(y_dat)
    ind = np.arange(len(y_dat))

    x_train, x_test, y_train, y_test, ind_train, ind_test = train_test_split(
        x_dat, y_dat_onehot, ind,
        test_size=0.25, random_state=42, shuffle=True, stratify=y_dat,
    )

    class_weights = util.class_weights(y_train[:, 1].astype("int"))

    model = mu.build_model(input_channels=n_channels)
    model.load_weights(initial_weights_path)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        batch_size=2048,
        epochs=400,
        class_weight=class_weights,
        validation_data=(x_test, y_test),
        callbacks=[callback],
    )

    model.save_weights(subdir_path / "trained_weights.h5")

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(subdir_path / "training_history.csv", index=False)

    class_predictions = model.predict(x_dat)
    predict_df = pd.DataFrame(class_predictions, columns=["prob_0", "prob_1"])
    predict_df["predicted_class"] = np.argmax(class_predictions, axis=1)
    predict_df["set"] = "train"
    predict_df.loc[ind_test, "set"] = "test"
    predict_df["date"] = param.dates.values
    predict_df["true_y"] = y_dat
    predict_df.to_csv(subdir_path / f"predicted_class_data_{region}_{subdir}.csv", index=False)

    return history, predict_df

x_dat = load_u_v_slp_input()

for region in REGIONS:
    print(f"Training U-V-SLP model — region: {region}")
    train_region(region, x_dat, subdir="U-V-SLP")
    print(f"Completed region: {region}")

x_dat = load_hgt_shum_slp_input()

for region in REGIONS:
    print(f"Training HGT-SHUM-SLP model — region: {region}")
    # NB: output subdirectory is named "Shum-SLP-HGT" (not "HGT-SHUM-SLP") to
    # match the folder naming already used for this combination's results.
    train_region(region, x_dat, subdir="Shum-SLP-HGT")
    print(f"Completed region: {region}")
