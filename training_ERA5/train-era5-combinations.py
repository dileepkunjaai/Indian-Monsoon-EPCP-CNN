from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from project_utils import read_utils as read
from project_utils import utils as util
from project_utils import parameters as param
from project_utils import model_utils as mu

# Reproducibility
np.random.seed(101)
tf.random.set_seed(333)

# Paths and constants
OUTPUT_DIR = Path("./processed_data")

SELECTED_LEVEL = 850
REGIONS = ['CNE', 'HR', 'NEI', 'NWI', 'IP', 'WCI']  # IMD homogeneous rainfall regions
BATCH_SIZE = 1024

def train_region(region, x_dat, output_suffix, output_dir=OUTPUT_DIR, batch_size=BATCH_SIZE):
    """
    Train and evaluate the CNN for a single IMD homogeneous rainfall region.

    Parameters
    ----------
    region : str
        Region abbreviation, e.g. "IP".
    x_dat : numpy.ndarray
        Predictor array for this variable combination, shape [time, lat, lon, vars].
    output_suffix : str
        Suffix identifying the variable combination in output file names
        (e.g. "gph_mslp", "uwind_vwind_slp", "hgt_slp_shum").
    output_dir : pathlib.Path
        Root directory for processed data and results.
    batch_size : int
        Training batch size.

    Returns
    -------
    (keras.callbacks.History, pandas.DataFrame)
        Training history and per-sample predictions.
    """
    region_path = output_dir / region

    y_dat = util.get_precip_classes(
        pd.read_csv(region_path / f"region_mean_precip_{region}.csv")["rain"],
        q=[0.95],
    )
    y_dat_onehot = util.onehot(y_dat)
    ind = np.arange(len(y_dat))

    # Guard against off-by-one length mismatches between predictors and labels
    min_length = min(len(x_dat), len(y_dat_onehot), len(ind))
    x_dat = x_dat[:min_length]
    y_dat_onehot = y_dat_onehot[:min_length]
    ind = ind[:min_length]
    y_dat = y_dat[:min_length]

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
        batch_size=batch_size,
        epochs=400,
        class_weight=class_weights,
        validation_data=(x_test, y_test),
        callbacks=[callback],
    )

    model.save_weights(region_path / f"trained_weights_{output_suffix}_{region}.h5")

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(region_path / f"training_history_{output_suffix}_{region}.csv", index=False)

    class_predictions = model.predict(x_dat)
    predict_df = pd.DataFrame(class_predictions, columns=["prob_0", "prob_1"])
    predict_df["predicted_class"] = np.argmax(class_predictions, axis=1)
    predict_df["set"] = "train"
    predict_df.loc[ind_test, "set"] = "test"

    # param.dates may not exactly match the (possibly truncated) sample count
    dates = param.dates.values[:len(predict_df)]
    if len(predict_df) > len(dates):
        predict_df = predict_df.iloc[:len(dates)]
    predict_df["date"] = dates
    predict_df["true_y"] = y_dat[:len(predict_df)]

    predict_df.to_csv(region_path / f"predicted_class_data_{output_suffix}_{region}.csv", index=False)

    return history, predict_df

x_dat = read.get_hgt_slp_input(SELECTED_LEVEL)

for region in REGIONS:
    print(f"Training GPH-MSLP model — region: {region}")
    train_region(region, x_dat, output_suffix="gph_mslp")
    print(f"Completed region: {region}")

x_dat = read.get_uwnd_vwnd_slp_input(SELECTED_LEVEL)

for region in REGIONS:
    print(f"Training U-V-SLP model — region: {region}")
    train_region(region, x_dat, output_suffix="uwind_vwind_slp")
    print(f"Completed region: {region}")

x_dat = read.get_hgt_slp_shum_input(SELECTED_LEVEL)

for region in REGIONS:
    print(f"Training HGT-SLP-SHUM model — region: {region}")
    train_region(region, x_dat, output_suffix="hgt_slp_shum")
    print(f"Completed region: {region}")
