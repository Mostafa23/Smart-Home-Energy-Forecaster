import subprocess, sys
for pkg in ["codecarbon"]:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                   check=False)

import os, json, joblib, warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (confusion_matrix, r2_score,
                                     mean_absolute_error,
                                     mean_absolute_percentage_error,
                                     roc_auc_score, f1_score,
                                     cohen_kappa_score,
                                     matthews_corrcoef,
                                     recall_score, precision_score,
                                     classification_report)
from tensorflow.keras.models    import Sequential, load_model
from tensorflow.keras.layers    import (Conv1D, MaxPooling1D, Flatten,
                                        Dense, Dropout,
                                        BatchNormalization, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils     import to_categorical

warnings.filterwarnings('ignore')

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

gpus = tf.config.list_physical_devices('GPU')
print(f"✅ GPUs available: {len(gpus)}")
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"✅ Using: {gpus[0]}")

BATCH_SIZE  = 512
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
print(f"✅ Batch size: {BATCH_SIZE}")

from codecarbon import EmissionsTracker
tracker = EmissionsTracker(project_name="smart_home_pipeline",
                           log_level="error",
                           save_to_file=False)
tracker.start()
print("✅ Carbon tracker started")

df_weather = pd.read_csv(
    '/kaggle/input/datasets/nikhil7280/weather-type-classification/'
    'weather_classification_data.csv'
)
df_weather.dropna(inplace=True)
print(f"\n✅ Dataset 2 raw: {df_weather.shape[0]:,} rows")

SHARED_FEATURES = [
    'Temperature',
    'Humidity',
    'Wind Speed',
    'Precipitation (%)',
    'Atmospheric Pressure',
    'Visibility (km)'
]

before = len(df_weather)
z      = np.abs(stats.zscore(df_weather[SHARED_FEATURES].astype(float)))
df_weather = df_weather[(z < 3).all(axis=1)].reset_index(drop=True)
print(f"   Outliers removed (Z>3): {before - len(df_weather):,} rows")
print(f"   After cleaning        : {len(df_weather):,} rows")
print(f"   Weather types         : {sorted(df_weather['Weather Type'].unique())}")

le_weather = LabelEncoder()
y_w_enc    = le_weather.fit_transform(df_weather['Weather Type'])
N_WEATHER  = len(le_weather.classes_)
print(f"   Classes ({N_WEATHER})            : {le_weather.classes_}")

X_w = df_weather[SHARED_FEATURES].values.astype(np.float32)

X_w_tr, X_w_te, yw_tr_idx, yw_te_idx = train_test_split(
    X_w, y_w_enc,
    test_size=0.2, random_state=RANDOM_SEED, stratify=y_w_enc
)

scaler_weather = StandardScaler()
X_w_tr_sc = scaler_weather.fit_transform(X_w_tr).astype(np.float32)
X_w_te_sc = scaler_weather.transform(X_w_te).astype(np.float32)

rng       = np.random.default_rng(RANDOM_SEED)
X_w_aug   = X_w_tr_sc + rng.normal(0, 0.02, X_w_tr_sc.shape).astype(np.float32)
X_w_tr_sc = np.vstack([X_w_tr_sc, X_w_aug])
yw_tr_idx = np.concatenate([yw_tr_idx, yw_tr_idx])

perm      = rng.permutation(len(X_w_tr_sc))
X_w_tr_sc = X_w_tr_sc[perm]
yw_tr_idx = yw_tr_idx[perm]

yw_tr_oh  = to_categorical(yw_tr_idx, N_WEATHER).astype(np.float32)
yw_te_oh  = to_categorical(yw_te_idx, N_WEATHER).astype(np.float32)

print(f"   Train after augmentation: {len(X_w_tr_sc):,} samples")
print(f"   Test  (clean)            : {len(X_w_te_sc):,} samples")

X_w_tr_cnn = X_w_tr_sc.reshape(-1, len(SHARED_FEATURES), 1)
X_w_te_cnn = X_w_te_sc.reshape(-1, len(SHARED_FEATURES), 1)

cnn = Sequential([
    Input(shape=(len(SHARED_FEATURES), 1)),
    Conv1D(64,  kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(N_WEATHER, activation='softmax')
], name='Weather_CNN')

cnn.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
cnn.summary()

es_cnn  = EarlyStopping(monitor='val_loss', patience=5,
                        restore_best_weights=True)
rlr_cnn = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                             patience=3, verbose=0)

print("\n--- Training CNN on Dataset 2 ---")
hist_cnn = cnn.fit(
    X_w_tr_cnn, yw_tr_oh,
    validation_split=0.15,
    epochs=40,
    batch_size=BATCH_SIZE,
    callbacks=[es_cnn, rlr_cnn],
    verbose=1
)

cnn_probs = cnn.predict(X_w_te_cnn, batch_size=BATCH_SIZE, verbose=0)
yw_pred   = np.argmax(cnn_probs, axis=1)
yw_true   = yw_te_idx

cnn_acc    = float(np.mean(yw_pred == yw_true))
cnn_auc    = float(roc_auc_score(yw_te_oh, cnn_probs, multi_class='ovr'))
cnn_f1     = float(f1_score(yw_true, yw_pred, average='macro'))
cnn_kappa  = float(cohen_kappa_score(yw_true, yw_pred))
cnn_mcc    = float(matthews_corrcoef(yw_true, yw_pred))
cnn_recall = float(recall_score(yw_true, yw_pred, average='macro'))
cnn_prec   = float(precision_score(yw_true, yw_pred, average='macro'))

print("\n" + "="*58)
print("📊  CNN CLASSIFICATION METRICS")
print(f"    Accuracy   : {cnn_acc:.4f}")
print(f"    ROC-AUC    : {cnn_auc:.4f}")
print(f"    F1 (macro) : {cnn_f1:.4f}")
print(f"    Kappa      : {cnn_kappa:.4f}")
print(f"    MCC        : {cnn_mcc:.4f}")
print(f"    Recall     : {cnn_recall:.4f}")
print(f"    Precision  : {cnn_prec:.4f}")
print("="*58)
print(classification_report(yw_true, yw_pred,
                             target_names=le_weather.classes_))

L0 = len(SHARED_FEATURES)
L1 = L0
L2 = L1 // 2
L3 = L2
L4 = L3 // 2
cnn_flops = (
    2 * L1 * 1   * 64  * 3 +
    2 * L3 * 64  * 128 * 3 +
    2 * L4 * 128 * 128     +
    2 * 128      * N_WEATHER
)
print(f"✅ CNN FLOPs (manual): {cnn_flops:,}")

cnn.save('weather_cnn_model.keras')
print("💾  weather_cnn_model.keras saved")

df_home = pd.read_csv(
    '/kaggle/input/datasets/taranvee/smart-home-dataset-with-weather-information/'
    'HomeC.csv',
    low_memory=False
)
df_home = df_home.loc[:, ~df_home.columns.str.contains('^Unnamed')]
df_home['time'] = pd.to_numeric(df_home['time'], errors='coerce')
df_home.dropna(subset=['time'], inplace=True)
df_home['time']      = pd.to_datetime(df_home['time'].astype(int), unit='s')
df_home              = df_home.sort_values('time').reset_index(drop=True)
df_home['hour']      = df_home['time'].dt.hour
df_home['month']     = df_home['time'].dt.month
df_home['dayofweek'] = df_home['time'].dt.dayofweek
df_home.drop(columns=['time'], inplace=True)
df_home.dropna(inplace=True)
print(f"\n✅ Dataset 1 raw: {df_home.shape[0]:,} rows")

before_h = len(df_home)
df_home  = df_home[df_home['use [kW]'].between(0.0, 20.0)].reset_index(drop=True)
print(f"   Energy outliers removed: {before_h - len(df_home):,} rows")

LEAKAGE_COLS = [
    'House overall [kW]', 'gen [kW]',        'Solar [kW]',
    'Dishwasher [kW]',    'Furnace 1 [kW]',  'Furnace 2 [kW]',
    'Home office [kW]',   'Fridge [kW]',     'Wine cellar [kW]',
    'Garage door [kW]',   'Kitchen 12 [kW]', 'Kitchen 14 [kW]',
    'Kitchen 38 [kW]',    'Barn [kW]',       'Well [kW]',
    'Microwave [kW]',     'Living room [kW]'
]
DROP_COLS    = ['use [kW]', 'icon', 'summary', 'cloudCover'] + LEAKAGE_COLS
feature_cols = [
    c for c in df_home.columns
    if c not in DROP_COLS
    and df_home[c].dtype in [np.float64, np.float32, np.int64, np.int32]
]
print(f"   Features before CNN injection: {len(feature_cols)}")

X_home_cnn_raw = np.column_stack([
    df_home['temperature'].values,
    df_home['humidity'].values          * 100,
    df_home['windSpeed'].values,
    df_home['precipProbability'].values * 100,
    df_home['pressure'].values,
    df_home['visibility'].values
]).astype(np.float32)

X_home_cnn_sc    = scaler_weather.transform(X_home_cnn_raw).astype(np.float32)
X_home_cnn_input = X_home_cnn_sc.reshape(-1, len(SHARED_FEATURES), 1)

print("\n--- CNN inferring weather labels for all HomeC rows ---")
weather_preds              = np.argmax(
    cnn.predict(X_home_cnn_input, batch_size=BATCH_SIZE * 2, verbose=1),
    axis=1
)
df_home['cnn_weather_type'] = weather_preds

print("\n✅ CNN weather distribution in HomeC:")
for lbl, cnt in (pd.Series(le_weather.inverse_transform(weather_preds))
                 .value_counts().items()):
    print(f"   {lbl:10s}: {cnt:,}")

feature_cols_mlp = feature_cols + ['cnn_weather_type']
print(f"\n✅ MLP feature count (leakage-free + CNN): {len(feature_cols_mlp)}")

X_mlp = df_home[feature_cols_mlp].values.astype(np.float32)
y_reg = df_home['use [kW]'].values.astype(np.float32)

split_idx  = int(len(df_home) * 0.80)
X_tr_raw   = X_mlp[:split_idx];  X_te_raw = X_mlp[split_idx:]
yr_train   = y_reg[:split_idx];  yr_test  = y_reg[split_idx:]
print(f"   Train (first 80% chronological): {len(X_tr_raw):,}")
print(f"   Test  (last  20% chronological): {len(X_te_raw):,}")

scaler_mlp  = StandardScaler()
X_tr_sc     = scaler_mlp.fit_transform(X_tr_raw).astype(np.float32)
X_te_sc     = scaler_mlp.transform(X_te_raw).astype(np.float32)

y_mean  = float(yr_train.mean())
y_std   = float(yr_train.std())
yr_tr_sc = ((yr_train - y_mean) / y_std).astype(np.float32)
yr_te_sc = ((yr_test  - y_mean) / y_std).astype(np.float32)
print(f"   Target — train mean: {y_mean:.4f} kW  std: {y_std:.4f} kW")
print(f"   Target — test  mean: {yr_test.mean():.4f} kW  "
      f"(drift: {yr_test.mean()-y_mean:+.4f} kW)")

X_tr_aug  = X_tr_sc + rng.normal(0, 0.01, X_tr_sc.shape).astype(np.float32)
yr_tr_aug = yr_tr_sc + rng.normal(0, 0.005, yr_tr_sc.shape).astype(np.float32)
X_tr_sc   = np.vstack([X_tr_sc,  X_tr_aug])
yr_tr_sc  = np.concatenate([yr_tr_sc, yr_tr_aug])
print(f"   Train after augmentation: {len(X_tr_sc):,} samples")

mlp = Sequential([
    Input(shape=(X_tr_sc.shape[1],)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(1)
], name='Energy_MLP')

mlp.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse', metrics=['mae'])
mlp.summary()

es_mlp  = EarlyStopping(monitor='val_loss', patience=7,
                        restore_best_weights=True)
rlr_mlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                             patience=3, min_lr=1e-6, verbose=1)

print("\n--- Training MLP on leakage-free HomeC + CNN weather feature ---")
hist_mlp = mlp.fit(
    X_tr_sc, yr_tr_sc,
    validation_split=0.15,
    epochs=40,
    batch_size=BATCH_SIZE,
    callbacks=[es_mlp, rlr_mlp],
    verbose=1
)

yr_pred_sc = mlp.predict(X_te_sc, batch_size=BATCH_SIZE * 2,
                          verbose=0).flatten()
yr_pred    = np.clip(yr_pred_sc * y_std + y_mean, 0.0, 20.0)

mlp_r2   = float(r2_score(yr_test, yr_pred))
mlp_mae  = float(mean_absolute_error(yr_test, yr_pred))
mlp_rmse = float(np.sqrt(np.mean((yr_test - yr_pred) ** 2)))
mask     = yr_test > 0.05
mlp_mape = float(mean_absolute_percentage_error(yr_test[mask], yr_pred[mask]))

print("\n" + "="*58)
print("📊  MLP REGRESSION METRICS  (time-based split, no leakage)")
print(f"    R²   : {mlp_r2:.4f}")
print(f"    MAE  : {mlp_mae:.4f} kW")
print(f"    RMSE : {mlp_rmse:.4f} kW")
print(f"    MAPE : {mlp_mape*100:.2f}%")
print("="*58)

n_feat    = X_tr_sc.shape[1]
mlp_flops = (
    2 * n_feat * 256 +
    2 * 256    * 128 +
    2 * 128    * 64  +
    2 * 64     * 1
)
print(f"✅ MLP FLOPs (manual): {mlp_flops:,}")

mlp.save('energy_mlp_model.keras')
print("💾  energy_mlp_model.keras saved")

try:
    co2_kg     = tracker.stop()
    energy_kwh = tracker._total_energy.kWh
    print(f"\n✅ Training energy : {energy_kwh:.6f} kWh")
    print(f"✅ Carbon emitted  : {co2_kg:.6f} kg CO₂eq")
except Exception as e:
    co2_kg = energy_kwh = -1
    print(f"⚠️  Carbon tracker: {e}")

fig, axes = plt.subplots(2, 4, figsize=(28, 12))
fig.suptitle(
    'Chained Pipeline: CNN (Weather Classification) → MLP (Energy Regression)',
    fontsize=16, fontweight='bold'
)

axes[0,0].plot(hist_cnn.history['accuracy'],     label='Train', lw=2)
axes[0,0].plot(hist_cnn.history['val_accuracy'], label='Val',   lw=2)
axes[0,0].set_title('CNN — Accuracy'); axes[0,0].set_xlabel('Epoch')
axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

axes[0,1].plot(hist_cnn.history['loss'],     label='Train', lw=2)
axes[0,1].plot(hist_cnn.history['val_loss'], label='Val',   lw=2)
axes[0,1].set_title('CNN — Loss (cross-entropy)'); axes[0,1].set_xlabel('Epoch')
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

cm = confusion_matrix(yw_true, yw_pred)
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le_weather.classes_,
            yticklabels=le_weather.classes_,
            ax=axes[0,2], cmap='Blues')
axes[0,2].set_title('CNN — Confusion Matrix')
axes[0,2].set_xlabel('Predicted'); axes[0,2].set_ylabel('True')

names_c = ['Accuracy','ROC-AUC','F1','Kappa','MCC','Recall','Precision']
vals_c  = [cnn_acc, cnn_auc, cnn_f1, cnn_kappa, cnn_mcc, cnn_recall, cnn_prec]
cols_c  = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860','#DA8BC3']
bars    = axes[0,3].bar(names_c, vals_c, color=cols_c, edgecolor='white')
axes[0,3].set_ylim(0, 1.12); axes[0,3].set_ylabel('Score')
axes[0,3].set_title('CNN — All Classification Metrics')
axes[0,3].tick_params(axis='x', rotation=30); axes[0,3].grid(alpha=0.3, axis='y')
for b, v in zip(bars, vals_c):
    axes[0,3].text(b.get_x()+b.get_width()/2, v+0.01,
                   f'{v:.3f}', ha='center', va='bottom', fontsize=8)

axes[1,0].plot(hist_mlp.history['loss'],     label='Train', lw=2)
axes[1,0].plot(hist_mlp.history['val_loss'], label='Val',   lw=2)
axes[1,0].set_title('MLP — Loss (MSE, scaled target)')
axes[1,0].set_xlabel('Epoch'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

axes[1,1].plot(hist_mlp.history['mae'],     label='Train', lw=2)
axes[1,1].plot(hist_mlp.history['val_mae'], label='Val',   lw=2)
axes[1,1].set_title('MLP — MAE (scaled units)')
axes[1,1].set_xlabel('Epoch'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

n_plot = min(3000, len(yr_test))
axes[1,2].scatter(yr_test[:n_plot], yr_pred[:n_plot],
                  alpha=0.2, s=3, color='steelblue')
lims = [min(yr_test.min(), yr_pred.min()), max(yr_test.max(), yr_pred.max())]
axes[1,2].plot(lims, lims, 'r--', lw=1.5, label='Perfect fit')
axes[1,2].set_title(f'MLP — Actual vs Predicted  (R²={mlp_r2:.4f})')
axes[1,2].set_xlabel('Actual (kW)'); axes[1,2].set_ylabel('Predicted (kW)')
axes[1,2].legend(); axes[1,2].grid(alpha=0.3)

residuals = yr_test - yr_pred
axes[1,3].hist(residuals, bins=60, color='steelblue',
               edgecolor='white', alpha=0.75)
axes[1,3].axvline(0, color='red', linestyle='--', lw=1.5)
axes[1,3].set_title('MLP — Residual Distribution')
axes[1,3].set_xlabel('Error (kW)'); axes[1,3].set_ylabel('Count')
axes[1,3].grid(alpha=0.3)
txt = (f"R²   = {mlp_r2:.4f}\n"
       f"MAE  = {mlp_mae:.4f} kW\n"
       f"RMSE = {mlp_rmse:.4f} kW\n"
       f"MAPE = {mlp_mape*100:.2f}%")
axes[1,3].text(0.97, 0.97, txt, transform=axes[1,3].transAxes,
               fontsize=9, va='top', ha='right',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                         edgecolor='gray', alpha=0.9))

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('chained_pipeline_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("💾  chained_pipeline_plots.png saved")

joblib.dump(scaler_weather,   'scaler_weather.pkl')
joblib.dump(scaler_mlp,       'scaler_mlp.pkl')
joblib.dump(le_weather,       'label_encoder_weather.pkl')
joblib.dump(feature_cols_mlp, 'feature_cols.pkl')
joblib.dump({'mean': y_mean, 'std': y_std}, 'target_scaler.pkl')
pd.DataFrame(hist_cnn.history).to_csv('cnn_history.csv', index=False)
pd.DataFrame(hist_mlp.history).to_csv('mlp_history.csv', index=False)

results = {
    "pipeline": "CNN (weather) → MLP (energy), chained",
    "preprocessing": {
        "outlier_removal"  : "Z-score < 3 on weather features; energy clipped [0,20] kW",
        "augmentation"     : "Gaussian noise std=0.02 (weather); std=0.01 (energy features)",
        "split_weather"    : "stratified random 80/20",
        "split_homec"      : "chronological 80/20 (time-based, no shuffle)",
        "target_scaling"   : f"StandardScaler on y_train (mean={y_mean:.4f}, std={y_std:.4f})",
        "leakage_fix"      : "17 sub-metering [kW] columns removed from MLP features"
    },
    "cnn_metrics": {
        "accuracy"         : round(cnn_acc,    4),
        "roc_auc_ovr"      : round(cnn_auc,    4),
        "f1_macro"         : round(cnn_f1,     4),
        "cohen_kappa"      : round(cnn_kappa,  4),
        "mcc"              : round(cnn_mcc,    4),
        "recall_macro"     : round(cnn_recall, 4),
        "precision_macro"  : round(cnn_prec,   4),
        "flops"            : cnn_flops
    },
    "mlp_metrics": {
        "r2"               : round(mlp_r2,   4),
        "mae_kw"           : round(mlp_mae,  4),
        "rmse_kw"          : round(mlp_rmse, 4),
        "mape_pct"         : round(mlp_mape * 100, 2),
        "flops"            : mlp_flops
    },
    "efficiency": {
        "training_energy_kwh": round(float(energy_kwh), 6) if energy_kwh != -1 else "N/A",
        "carbon_kg_co2eq"    : round(float(co2_kg),     6) if co2_kg     != -1 else "N/A"
    }
}
with open('final_evaluation.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n" + "="*62)
print("💾  ALL ARTIFACTS SAVED")
print("    weather_cnn_model.keras   |  energy_mlp_model.keras")
print("    scaler_weather.pkl        |  scaler_mlp.pkl")
print("    label_encoder_weather.pkl |  feature_cols.pkl")
print("    target_scaler.pkl         |  final_evaluation.json")
print("    cnn_history.csv           |  mlp_history.csv")
print("    chained_pipeline_plots.png")
print("="*62)
print("\n📊  CNN CLASSIFICATION METRICS")
print(f"    Accuracy   : {cnn_acc:.4f}")
print(f"    ROC-AUC    : {cnn_auc:.4f}")
print(f"    F1 (macro) : {cnn_f1:.4f}")
print(f"    Kappa      : {cnn_kappa:.4f}")
print(f"    MCC        : {cnn_mcc:.4f}")
print(f"    Recall     : {cnn_recall:.4f}")
print(f"    Precision  : {cnn_prec:.4f}")
print(f"    FLOPs      : {cnn_flops:,}")
print("\n📊  MLP REGRESSION METRICS")
print(f"    R²         : {mlp_r2:.4f}")
print(f"    MAE        : {mlp_mae:.4f} kW")
print(f"    RMSE       : {mlp_rmse:.4f} kW")
print(f"    MAPE       : {mlp_mape*100:.2f}%")
print(f"    FLOPs      : {mlp_flops:,}")
print("\n⚡  EFFICIENCY")
if energy_kwh != -1:
    print(f"    Training energy : {energy_kwh:.6f} kWh")
    print(f"    Carbon emitted  : {co2_kg:.6f} kg CO₂eq")
print("="*62)

def predict_smart_home(raw_reading: dict):
    m_cnn = load_model('weather_cnn_model.keras', compile=False)
    m_mlp = load_model('energy_mlp_model.keras',  compile=False)
    sc_w  = joblib.load('scaler_weather.pkl')
    sc_m  = joblib.load('scaler_mlp.pkl')
    enc   = joblib.load('label_encoder_weather.pkl')
    fcols = joblib.load('feature_cols.pkl')
    tsc   = joblib.load('target_scaler.pkl')

    cnn_in = np.array([[
        raw_reading['temperature'],
        raw_reading['humidity']          * 100,
        raw_reading['windSpeed'],
        raw_reading['precipProbability'] * 100,
        raw_reading['pressure'],
        raw_reading['visibility']
    ]], dtype=np.float32)
    cnn_scaled    = sc_w.transform(cnn_in).reshape(1, 6, 1)
    weather_idx   = int(np.argmax(m_cnn.predict(cnn_scaled, verbose=0)))
    weather_label = enc.inverse_transform([weather_idx])[0]

    raw_reading['cnn_weather_type'] = weather_idx
    mlp_in      = np.array([[raw_reading.get(c, 0.0) for c in fcols]],
                            dtype=np.float32)
    pred_scaled = float(m_mlp.predict(sc_m.transform(mlp_in), verbose=0)[0][0])
    energy_kw   = float(np.clip(pred_scaled * tsc['std'] + tsc['mean'], 0.0, 20.0))

    print("\n" + "="*55)
    print("📊  CHAINED PIPELINE PREDICTION")
    print(f"    CNN  → Weather : {weather_label}")
    print(f"    MLP  → Energy  : {energy_kw:.4f} kW")
    print("="*55)
    return weather_label, energy_kw

test_sample = df_home.iloc[0].to_dict()
predict_smart_home(test_sample)
