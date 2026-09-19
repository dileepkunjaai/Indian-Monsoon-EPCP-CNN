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
OUTPUT_DIR = Path("./processed_data")

HGT_850_FILE = DATA_DIR / "hgt_anomalies_850-jjas-22.nc"
SLP_FILE = DATA_DIR / "slp_anomalies-jjas-22.nc"

REGIONS = ["HR", "NEI", "NWI", "WCI", "IP", "CNE"]  # IMD homogeneous rainfall regions
SUBDIR = "850hpa"  # 850 hPa geopotential height + SLP combination

def load_hgt850_slp_input(hgt_path=HGT_850_FILE, slp_path=SLP_FILE):
    """
    Build the CNN predictor array from 850 hPa geopotential height and SLP anomalies.

    Parameters
    ----------
    hgt_path : path-like
        Path to the 850 hPa geopotential height anomaly file.
    slp_path : path-like
        Path to the SLP anomaly file.

    Returns
    -------
    numpy.ndarray
        Array of shape [time, lat, lon, 2], channel order (hgt850, slp).
    """
    hgt_xr = xr.open_dataset(hgt_path)
    hgt_data = np.array(hgt_xr["hgt_anom"]).reshape(param.n, param.nlats, param.nlons, 1)

    slp_xr = xr.open_dataset(slp_path)
    slp_data = np.array(slp_xr["slp_anom"]).reshape(param.n, param.nlats, param.nlons, 1)

    return np.concatenate([hgt_data, slp_data], axis=3)

def train_region(region, x_dat, subdir=SUBDIR, output_dir=OUTPUT_DIR):
    """
    Train and evaluate the CNN for a single IMD homogeneous rainfall region.

    Parameters
    ----------
    region : str
        Region abbreviation, e.g. "IP".
    x_dat : numpy.ndarray
        Predictor array from `load_hgt850_slp_input`.
    subdir : str
        Output subdirectory for this variable combination.
    output_dir : pathlib.Path
        Root directory for processed data and results.

    Returns
    -------
    (keras.callbacks.History, pandas.DataFrame)
        Training history and per-sample predictions.
    """
    region_path = output_dir / region
    subdir_path = region_path / subdir
    subdir_path.mkdir(parents=True, exist_ok=True)

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

    model = mu.build_model()
    model.load_weights(output_dir / "initial_weights.h5")

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
    hist_df.to_csv(subdir_path / "training_history.csv")

    class_predictions = model.predict(x_dat)
    predict_df = pd.DataFrame(class_predictions).rename(columns={0: "prob_0", 1: "prob_1"})
    predict_df["predicted_class"] = np.argmax(class_predictions, axis=1)
    predict_df["set"] = "train"
    predict_df.loc[ind_test, "set"] = "test"
    predict_df["date"] = param.dates.values
    predict_df["true_y"] = y_dat
    predict_df.to_csv(subdir_path / f"predicted_class_data_{region}_{subdir}.csv", index=False)

    return history, predict_df

x_dat = load_hgt850_slp_input()

results = {}
for region in REGIONS:
    print(f"Training region: {region}")
    results[region] = train_region(region, x_dat)
    print(f"Completed region: {region}")
