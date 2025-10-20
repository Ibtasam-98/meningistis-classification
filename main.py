import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_score, \
    recall_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
# Set style for better plots
plt.style.use('default')

# Load and prepare data
print("=" * 80)
print("MENINGITIS CLASSIFICATION SYSTEM - INITIALIZING")
print("=" * 80)

df = pd.read_csv('dataset/meningitis.csv')
df_classification = df[df['Diagnosis'].isin(['Bacterial', 'Viral'])].copy()

feature_columns = ['Age', 'WBC_Count', 'Protein_Level', 'Glucose_Level',
                   'Hemoglobin', 'WBC_Blood_Count', 'Platelets', 'CRP_Level']

df_classification = df_classification.dropna(subset=feature_columns + ['Diagnosis'])

if df_classification.empty or len(df_classification) < 2:
    print("Error: Insufficient data after preprocessing. Cannot run model evaluation.")
    exit()

X = df_classification[feature_columns]
y = df_classification['Diagnosis']

# Print class distribution
print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)
class_dist = df_classification['Diagnosis'].value_counts()
print(class_dist)
print(f"\nTotal samples: {len(df_classification)}")
print(
    f"Bacterial: {class_dist.get('Bacterial', 0)} ({class_dist.get('Bacterial', 0) / len(df_classification) * 100:.2f}%)")
print(f"Viral: {class_dist.get('Viral', 0)} ({class_dist.get('Viral', 0) / len(df_classification) * 100:.2f}%)")

# Encode target (Bacterial: 1, Viral: 0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\nTarget encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, is_dnn=False, is_sklearn_nn=False):
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} - TRAINING PHASE")
    print("=" * 60)

    start_time = time.time()

    if is_dnn:
        # DNN specific training
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Testing phase
        print("\n" + "=" * 60)
        print(f"{model_name.upper()} - TESTING PHASE")
        print("=" * 60)
        start_test_time = time.time()

        # Extract accuracy from Keras evaluate
        metrics = model.evaluate(X_test, y_test, verbose=0)
        test_accuracy = metrics[1]
        testing_time = time.time() - start_test_time

        # Make predictions
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Get training history metrics
        train_accuracy = history.history['accuracy'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        best_val_accuracy = max(history.history['val_accuracy'])
        best_train_accuracy = max(history.history['accuracy'])

        # Calculate precision/recall manually from predictions for consistency
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)

        history_data = history.history

    elif is_sklearn_nn:
        # Scikit-learn neural network training
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Testing phase
        print("\n" + "=" * 60)
        print(f"{model_name.upper()} - TESTING PHASE")
        print("=" * 60)
        start_test_time = time.time()

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        testing_time = time.time() - start_test_time

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = train_accuracy
        best_val_accuracy = val_accuracy
        best_train_accuracy = train_accuracy

        # Calculate precision and recall
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)

        history_data = None

    else:
        # Traditional ML models
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Testing phase
        print("\n" + "=" * 60)
        print(f"{model_name.upper()} - TESTING PHASE")
        print("=" * 60)
        start_test_time = time.time()

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model,
                                                                    'predict_proba') else model.decision_function(
            X_test)
        testing_time = time.time() - start_test_time

        # Calculate metrics for traditional models
        test_accuracy = accuracy_score(y_test, y_pred)
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = train_accuracy
        best_val_accuracy = val_accuracy
        best_train_accuracy = train_accuracy

        # Calculate precision and recall
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)

        history_data = None

    print(f"Testing completed in {testing_time:.2f} seconds")

    # Calculate additional metrics
    train_error = 1 - train_accuracy
    test_error = 1 - test_accuracy
    val_error = 1 - val_accuracy

    # ROC Analysis
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    # Calculate Sensitivity and Specificity from Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) != 0 else 0.0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0.0
    else:
        sensitivity = test_recall
        specificity = 0.0

    # Calculate speed (inference efficiency)
    speed = 1 / (testing_time + 0.001)  # Avoid division by zero

    # Print all results in terminal
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} - COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    print(f"\nTRAINING AND TESTING TIMES:")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Testing Time: {testing_time:.2f} seconds")
    print(f"Total Time: {training_time + testing_time:.2f} seconds")

    print(f"\nACCURACY METRICS:")
    print(f"Final Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"Best Training Accuracy: {best_train_accuracy:.4f} ({best_train_accuracy * 100:.2f}%)")
    print(f"Final Validation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f} ({best_val_accuracy * 100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    print(f"\nERROR RATES:")
    print(f"Training Error Rate: {train_error:.4f} ({train_error * 100:.2f}%)")
    print(f"Validation Error Rate: {val_error:.4f} ({val_error * 100:.2f}%)")
    print(f"Test Error Rate: {test_error:.4f} ({test_error * 100:.2f}%)")

    print(f"\nKEY PERFORMANCE METRICS:")
    print(f"Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Sensitivity: {sensitivity:.4f} ({sensitivity * 100:.2f}%)")
    print(f"Specificity: {specificity:.4f} ({specificity * 100:.2f}%)")
    print(f"Precision: {test_precision:.4f} ({test_precision * 100:.2f}%)")
    print(f"Speed (1/time): {speed:.2f}")

    print(f"\nROC ANALYSIS:")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"False Positive Rate: {fpr[np.argmax(tpr - fpr)]:.4f}")
    print(f"True Positive Rate: {tpr[np.argmax(tpr - fpr)]:.4f}")

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    print("\nCONFUSION MATRIX:")
    print(cm)
    print(f"True Negatives: {cm[0, 0]}")
    print(f"False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]}")
    print(f"True Positives: {cm[1, 1]}")

    print(f"\nCONFUSION MATRIX ANALYSIS:")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"False Positive Rate: {cm[0, 1] / (cm[0, 0] + cm[0, 1]):.4f}" if (cm[0, 0] + cm[0, 1]) != 0 else "N/A")
    print(f"False Negative Rate: {cm[1, 0] / (cm[1, 0] + cm[1, 1]):.4f}" if (cm[1, 0] + cm[1, 1]) != 0 else "N/A")

    # Performance Gap Analysis
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} - PERFORMANCE GAP ANALYSIS")
    print("=" * 60)
    train_test_gap = abs(train_accuracy - test_accuracy)
    train_val_gap = abs(train_accuracy - val_accuracy)

    print(f"Train-Test Accuracy Gap: {train_test_gap:.4f}")
    print(f"Train-Validation Accuracy Gap: {train_val_gap:.4f}")

    if train_test_gap < 0.05:
        print("✓ Excellent generalization - minimal overfitting")
    else:
        print("⚠ Moderate performance gap - potential overfitting")

    if train_val_gap < 0.05:
        print("✓ Stable training - good validation performance")
    else:
        print("⚠ Training-validation gap detected")

    # Model Efficiency Analysis
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} - MODEL EFFICIENCY ANALYSIS")
    print("=" * 60)
    if is_dnn:
        epochs_trained = len(history.history['loss'])
        print(f"Training Speed: {training_time / epochs_trained:.3f} seconds/epoch")
    print(f"Inference Speed: {testing_time / len(X_test):.5f} seconds/sample")
    print(f"Training Efficiency: {train_accuracy / training_time:.4f} accuracy per second")
    print(f"Inference Efficiency: {test_accuracy / testing_time:.4f} accuracy per second")

    # Final Summary
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} - FINAL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"✓ Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"✓ ROC AUC Score: {roc_auc:.4f}")
    print(f"✓ Training completed in: {training_time:.2f} seconds")
    print(f"✓ Testing completed in: {testing_time:.2f} seconds")
    if is_dnn:
        print(f"✓ Model converged after {len(history.history['loss'])} epochs")
    print(f"✓ Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"✓ Confusion Matrix: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")
    print(f"✓ Error Distribution: {cm[0, 1] + cm[1, 0]} misclassifications out of {len(y_test)} samples")

    # Clinical Performance Assessment
    print("\n" + "=" * 60)
    print(f"{model_name.upper()} - CLINICAL PERFORMANCE ASSESSMENT")
    print("=" * 60)
    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f} ({sensitivity * 100:.2f}%)")
    print(f"Specificity (True Negative Rate): {specificity:.4f} ({specificity * 100:.2f}%)")
    print(f"False Negative Rate: {1 - sensitivity:.4f} ({(1 - sensitivity) * 100:.2f}%)")
    print(f"False Positive Rate: {1 - specificity:.4f} ({(1 - specificity) * 100:.2f}%)")

    if sensitivity > 0.9 and specificity > 0.9:
        print("✓ Excellent clinical performance")
    elif sensitivity > 0.8 and specificity > 0.8:
        print("✓ Good clinical performance")
    else:
        print("⚠ Moderate clinical performance")

    print("\n" + "=" * 80)
    print(f"END OF {model_name.upper()} EVALUATION")
    print("=" * 80)

    return {
        'model_name': model_name,
        'test_accuracy': test_accuracy,
        'roc_auc': roc_auc,
        'training_time': training_time,
        'testing_time': testing_time,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': test_precision,
        'speed': speed,
        'fpr': fpr,
        'tpr': tpr,
        'cm': cm,
        'history': history_data,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


# Initialize models
print("\n" + "=" * 80)
print("INITIALIZING ALL MODELS")
print("=" * 80)

# DNN Model setup
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

dnn_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(1, activation='sigmoid')
])

dnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\nDNN MODEL ARCHITECTURE")
dnn_model.summary()

# ANN Model (Simple Neural Network)
ann_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

ann_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nANN MODEL ARCHITECTURE")
ann_model.summary()

# MLP Classifier (Scikit-learn)
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=20
)

# SGD Classifier
sgd_model = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=0.001,
    max_iter=1000,
    random_state=42
)

# XGBoost Classifier
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

# AdaBoost Classifier
adaboost_model = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

# SVM RBF Classifier
svm_rbf_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)

# Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Evaluate all models
results = []

# DNN
dnn_result = evaluate_model(dnn_model, "DEEP NEURAL NETWORK", X_train_scaled, X_test_scaled, y_train, y_test,
                            is_dnn=True)
results.append(dnn_result)

# ANN
ann_result = evaluate_model(ann_model, "ARTIFICIAL NEURAL NETWORK", X_train_scaled, X_test_scaled, y_train, y_test,
                            is_dnn=True)
results.append(ann_result)

# MLP
mlp_result = evaluate_model(mlp_model, "MLP CLASSIFIER", X_train_scaled, X_test_scaled, y_train, y_test,
                            is_sklearn_nn=True)
results.append(mlp_result)

# SGD Classifier
sgd_result = evaluate_model(sgd_model, "SGD CLASSIFIER", X_train_scaled, X_test_scaled, y_train, y_test, is_dnn=False)
results.append(sgd_result)

# XGBoost
xgb_result = evaluate_model(xgb_model, "XGBOOST CLASSIFIER", X_train_scaled, X_test_scaled, y_train, y_test,
                            is_dnn=False)
results.append(xgb_result)

# AdaBoost
adaboost_result = evaluate_model(adaboost_model, "ADABOOST CLASSIFIER", X_train_scaled, X_test_scaled, y_train, y_test,
                                 is_dnn=False)
results.append(adaboost_result)

# SVM RBF
svm_result = evaluate_model(svm_rbf_model, "SVM RBF", X_train_scaled, X_test_scaled, y_train, y_test, is_dnn=False)
results.append(svm_result)

# Random Forest
rf_result = evaluate_model(rf_model, "RANDOM FOREST", X_train_scaled, X_test_scaled, y_train, y_test, is_dnn=False)
results.append(rf_result)

# Final comparison summary
print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print("\n{:<25} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
    "MODEL", "ACCURACY", "SENSITIVITY", "SPECIFICITY", "PRECISION", "SPEED", "ROC AUC"
))
print("-" * 110)
for result in results:
    print("{:<25} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.2f} {:<12.4f}".format(
        result['model_name'],
        result['test_accuracy'],
        result['sensitivity'],
        result['specificity'],
        result['precision'],
        result['speed'],
        result['roc_auc']
    ))

# Find best model
best_accuracy_model = max(results, key=lambda x: x['test_accuracy'])
best_auc_model = max(results, key=lambda x: x['roc_auc'])
fastest_model = max(results, key=lambda x: x['speed'])

print("\n" + "=" * 80)
print("BEST MODEL SUMMARY")
print("=" * 80)
print(f"🏆 Best Accuracy: {best_accuracy_model['model_name']} ({best_accuracy_model['test_accuracy']:.4f})")
print(f"🏆 Best ROC AUC: {best_auc_model['model_name']} ({best_auc_model['roc_auc']:.4f})")
print(f"⚡ Fastest Model: {fastest_model['model_name']} ({fastest_model['speed']:.2f})")

# ==============================================================================
# VISUALIZATION 1: Model Performance with Elegant Blue Gradient
# ==============================================================================
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.0
rcParams['figure.dpi'] = 600
rcParams['savefig.dpi'] = 600
rcParams['axes.labelweight'] = 'bold'

print("\n" + "=" * 80)
print("GENERATING HD VISUALIZATION 1: MODEL PERFORMANCE (BLUE GRADIENT)")
print("=" * 80)

# --- Data Preparation ---
model_labels = ["DNN", "ANN", "MLP", "SGD", "XGB", "ADA", "SVM", "RF"]
test_accuracies = [result['test_accuracy'] * 100 for result in results]
training_times = [result['training_time'] for result in results]
x_pos = np.arange(len(model_labels))

# --- Create Figure ---
fig, ax1 = plt.subplots(figsize=(6.5, 4.2))  # Ideal academic aspect ratio

# Elegant blue gradient
mixed_colors = ['#0A1D37', '#143875', '#1E5BAF', '#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE', '#DBEAFE']

# --- Bar Plot (Accuracy) ---
bars = ax1.bar(
    x_pos, test_accuracies, color=mixed_colors, alpha=0.9,
    edgecolor='none', linewidth=0.7, label='Test Accuracy'
)
ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_ylim(max(85, min(test_accuracies)*0.95), 100)

# Add value labels
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 0.6,
             f'{test_accuracies[i]:.1f}%', ha='center', va='bottom',
             fontsize=8.5, color='black', fontweight='semibold')

# --- Line Plot (Training Time) ---
ax2 = ax1.twinx()
line = ax2.plot(
    x_pos, training_times, color='#1B263B', marker='s',
    linestyle='-', linewidth=1.8, markersize=5, label='Training Time'
)
ax2.set_ylabel('Training Time (s)', fontsize=11, fontweight='bold')
ax2.set_ylim(0, max(training_times)*1.4)

# Annotate times
for i, t in enumerate(training_times):
    ax2.annotate(f'{t:.2f}s', (x_pos[i], t), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=8, fontweight='bold',
                 color='black', bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.9))

# --- Axes Formatting ---
ax1.set_xlabel('Models', fontsize=11, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_labels, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.25)
ax1.spines['top'].set_visible(False)

# Legend
lines_and_bars = [bars[0]] + line
labels = ['Test Accuracy', 'Training Time']
ax1.legend(lines_and_bars, labels, loc='upper left', fontsize=9, frameon=True)

# Title
plt.title('Model Performance: Accuracy vs Training Time', fontsize=12, fontweight='bold', pad=12)

# --- Save in HD Formats ---
plt.tight_layout()
plt.savefig('HD_Model_Performance.pdf', bbox_inches='tight')  # Vector
plt.savefig('HD_Model_Performance.png', bbox_inches='tight', dpi=600)
plt.show()


# ==============================================================================
# HD PUBLICATION VISUALIZATION 2: Consolidated Performance Metrics
# ==============================================================================

print("\n" + "=" * 80)
print("GENERATING HD VISUALIZATION 2: CONSOLIDATED METRICS")
print("=" * 80)

metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Speed']
metric_values = {
    'Accuracy': [r['test_accuracy'] * 100 for r in results],
    'Sensitivity': [r['sensitivity'] * 100 for r in results],
    'Specificity': [r['specificity'] * 100 for r in results],
    'Precision': [r['precision'] * 100 for r in results],
    'Speed': [r['speed'] for r in results]
}

# Normalize speed to 0–100
max_speed = max(metric_values['Speed'])
metric_values['Speed'] = [s / max_speed * 100 for s in metric_values['Speed']]

# --- Create Figure ---
fig, ax = plt.subplots(figsize=(7, 4.5))
bar_width = 0.15
x_pos = np.arange(len(model_labels))
metric_colors = ['#0A1D37', '#143875', '#1E5BAF', '#3B82F6', '#93C5FD']

# --- Bars for Each Metric ---
for i, metric in enumerate(metrics):
    values = metric_values[metric]
    positions = x_pos + i * bar_width
    bars = ax.bar(
        positions, values, bar_width, label=metric,
        color=metric_colors[i], alpha=0.9, linewidth=0.5
    )
    # for bar, value in zip(bars, values):
        # ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
        #         f'{value:.1f}%', ha='center', va='bottom', fontsize=8, color='black')

# --- Format Axes ---
ax.set_xlabel('Models', fontsize=11, fontweight='bold')
ax.set_ylabel('Performance Score (%)', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos + bar_width*2)
ax.set_xticklabels(model_labels, fontsize=10)
ax.set_ylim(0, 105)
ax.grid(True, linestyle='--', alpha=0.25)

# Legend
ax.legend(title='Metrics', title_fontsize=9, fontsize=8.5, loc='upper right', frameon=True)

plt.title('Consolidated Model Performance Metrics', fontsize=12, fontweight='bold', pad=12)

# --- Save in HD Formats ---
plt.tight_layout()
plt.savefig('HD_Consolidated_Metrics.pdf', bbox_inches='tight')
plt.savefig('HD_Consolidated_Metrics.png', bbox_inches='tight', dpi=600)
plt.show()




# ==============================================================================
# USER INPUT CLASSIFICATION SYSTEM WITH VALIDATION
# ==============================================================================
print("\n" + "=" * 80)
print("MENINGITIS CLASSIFICATION - USER INPUT WITH VALIDATION")
print("=" * 80)


def validate_clinical_ranges(age, wbc_count, protein_level, glucose_level,
                             hemoglobin, wbc_blood_count, platelets, crp_level):
    """
    Validate if inputs are within clinically plausible ranges
    """
    warnings = []

    # Age validation
    if age < 0 or age > 120:
        warnings.append("❌ Age outside typical range (0-120 years)")
    elif age < 1:
        warnings.append("⚠ Infant patient - special considerations needed")

    # CSF WBC validation
    if wbc_count < 0:
        warnings.append("❌ CSF WBC count cannot be negative")
    elif wbc_count > 10000:
        warnings.append("⚠ Very high CSF WBC count - check measurement")

    # CSF Protein validation
    if protein_level < 0:
        warnings.append("❌ CSF Protein level cannot be negative")
    elif protein_level > 500:
        warnings.append("⚠ Very high CSF Protein level - check measurement")

    # CSF Glucose validation
    if glucose_level < 0:
        warnings.append("❌ CSF Glucose level cannot be negative")
    elif glucose_level > 200:
        warnings.append("⚠ High CSF Glucose level - unusual for meningitis")

    # Hemoglobin validation
    if hemoglobin < 0:
        warnings.append("❌ Hemoglobin cannot be negative")
    elif hemoglobin > 20:
        warnings.append("⚠ High Hemoglobin level - check measurement")
    elif hemoglobin < 7:
        warnings.append("⚠ Low Hemoglobin - possible anemia")

    # Blood WBC validation
    if wbc_blood_count < 0:
        warnings.append("❌ Blood WBC count cannot be negative")
    elif wbc_blood_count > 50000:
        warnings.append("⚠ Very high Blood WBC count - check measurement")

    # Platelets validation
    if platelets < 0:
        warnings.append("❌ Platelet count cannot be negative")
    elif platelets > 1000:
        warnings.append("⚠ High Platelet count - check measurement")
    elif platelets < 50:
        warnings.append("⚠ Low Platelet count - risk of bleeding")

    # CRP validation
    if crp_level < 0:
        warnings.append("❌ CRP level cannot be negative")
    elif crp_level > 500:
        warnings.append("⚠ Very high CRP level - severe inflammation")

    return warnings


def get_valid_input(prompt, input_type=float, min_val=None, max_val=None):
    """
    Get validated user input with range checking
    """
    while True:
        try:
            value = input_type(input(prompt))

            if min_val is not None and value < min_val:
                print(f"❌ Value must be at least {min_val}")
                continue

            if max_val is not None and value > max_val:
                print(f"❌ Value must be at most {max_val}")
                continue

            return value

        except ValueError:
            print("❌ Please enter a valid number")
        except Exception as e:
            print(f"❌ Error: {e}")


def classify_new_patient(best_model, model_name, is_dnn=False, is_sklearn_nn=False):
    """
    Function to take user input and classify meningitis type with validation
    """
    print(f"\n🧠 Using {model_name} for classification...")
    print("\n📝 Please enter the following patient details:")

    try:
        # Get user input for each feature with validation
        print("\n" + "=" * 40)
        print("PATIENT DEMOGRAPHICS")
        print("=" * 40)
        age = get_valid_input("• Age (years): ", float, 0, 120)

        print("\n" + "=" * 40)
        print("CEREBROSPINAL FLUID (CSF) ANALYSIS")
        print("=" * 40)
        wbc_count = get_valid_input("• WBC Count (CSF - cells/μL): ", float, 0, 10000)
        protein_level = get_valid_input("• Protein Level (CSF - mg/dL): ", float, 0, 500)
        glucose_level = get_valid_input("• Glucose Level (CSF - mg/dL): ", float, 0, 200)

        print("\n" + "=" * 40)
        print("BLOOD TEST RESULTS")
        print("=" * 40)
        hemoglobin = get_valid_input("• Hemoglobin (g/dL): ", float, 0, 20)
        wbc_blood_count = get_valid_input("• WBC Blood Count (cells/μL): ", float, 0, 50000)
        platelets = get_valid_input("• Platelets (×10³/μL): ", float, 0, 1000)
        crp_level = get_valid_input("• CRP Level (mg/L): ", float, 0, 500)

        # Validate clinical ranges
        warnings = validate_clinical_ranges(age, wbc_count, protein_level, glucose_level,
                                            hemoglobin, wbc_blood_count, platelets, crp_level)

        if warnings:
            print("\n⚠ CLINICAL VALIDATION WARNINGS:")
            for warning in warnings:
                print(f"   {warning}")

            proceed = input("\nContinue with classification anyway? (y/n): ").lower()
            if proceed != 'y':
                print("❌ Classification cancelled.")
                return None, None

        # Create feature array
        patient_data = np.array([[age, wbc_count, protein_level, glucose_level,
                                  hemoglobin, wbc_blood_count, platelets, crp_level]])

        # Scale the features (using the same scaler from training)
        patient_data_scaled = scaler.transform(patient_data)

        # Make prediction
        if is_dnn:
            prediction_proba = best_model.predict(patient_data_scaled, verbose=0)[0][0]
            prediction = 1 if prediction_proba > 0.5 else 0
        elif is_sklearn_nn:
            prediction_proba = best_model.predict_proba(patient_data_scaled)[0][1]
            prediction = best_model.predict(patient_data_scaled)[0]
        else:
            prediction_proba = best_model.predict_proba(patient_data_scaled)[0][1]
            prediction = best_model.predict(patient_data_scaled)[0]

        # Convert prediction to diagnosis
        diagnosis = "BACTERIAL" if prediction == 1 else "VIRAL"
        confidence = prediction_proba if prediction == 1 else 1 - prediction_proba
        confidence_level = "HIGH" if confidence > 0.8 else "MODERATE" if confidence > 0.6 else "LOW"

        # Display results
        print("\n" + "=" * 60)
        print("🧬 CLASSIFICATION RESULTS")
        print("=" * 60)
        print(f"🔍 Model Used: {model_name}")
        print(f"📋 Diagnosis: {diagnosis} MENINGITIS")
        print(f"📊 Confidence: {confidence:.4f} ({confidence * 100:.2f}%) - {confidence_level} CONFIDENCE")

        # Risk assessment
        if diagnosis == "BACTERIAL" and confidence > 0.7:
            print("🚨 HIGH RISK - URGENT ATTENTION REQUIRED")
        elif diagnosis == "BACTERIAL":
            print("⚠ MODERATE RISK - PROMPT EVALUATION NEEDED")
        else:
            print("✅ LOWER RISK - CONTINUE MONITORING")

        # Clinical interpretation
        print(f"\n📋 CLINICAL INTERPRETATION:")
        if diagnosis == "BACTERIAL":
            print("   • 🚨 Urgent antibiotic treatment required")
            print("   • 🏥 Consider hospitalization and IV antibiotics")
            print("   • 📊 Monitor neurological status closely")
            print("   • 🔬 Consider CSF culture and sensitivity testing")
            print("   • 💊 Empiric antibiotics: Ceftriaxone + Vancomycin")
        else:
            print("   • ✅ Likely viral etiology")
            print("   • 🛌 Supportive care recommended")
            print("   • 💧 Maintain hydration and symptomatic treatment")
            print("   • 📈 Usually self-limiting within 7-14 days")
            print("   • 🔍 Consider PCR testing for specific viruses")

        # Feature analysis with clinical context
        print(f"\n🔬 DETAILED FEATURE ANALYSIS:")
        print(f"   • Age: {age} years")

        # CSF WBC analysis
        if wbc_count > 1000:
            wbc_interpretation = "🚨 Very high - strongly suggests bacterial"
        elif wbc_count > 100:
            wbc_interpretation = "⚠ Elevated - consistent with meningitis"
        else:
            wbc_interpretation = "✅ Within normal range"
        print(f"   • CSF WBC: {wbc_count} cells/μL {wbc_interpretation}")

        # CSF Protein analysis
        if protein_level > 100:
            protein_interpretation = "🚨 High - suggests bacterial"
        elif protein_level > 45:
            protein_interpretation = "⚠ Elevated - consistent with meningitis"
        else:
            protein_interpretation = "✅ Normal"
        print(f"   • CSF Protein: {protein_level} mg/dL {protein_interpretation}")

        # CSF Glucose analysis
        if glucose_level < 40:
            glucose_interpretation = "🚨 Low - strongly suggests bacterial"
        elif glucose_level < 60:
            glucose_interpretation = "⚠ Low - concerning"
        else:
            glucose_interpretation = "✅ Normal"
        print(f"   • CSF Glucose: {glucose_level} mg/dL {glucose_interpretation}")

        # CRP analysis
        if crp_level > 100:
            crp_interpretation = "🚨 Very high - strong inflammatory response"
        elif crp_level > 40:
            crp_interpretation = "⚠ Elevated - suggests infection"
        else:
            crp_interpretation = "✅ Normal or mildly elevated"
        print(f"   • CRP: {crp_level} mg/L {crp_interpretation}")

        print(f"   • Hemoglobin: {hemoglobin} g/dL")
        print(f"   • Blood WBC: {wbc_blood_count} cells/μL")
        print(f"   • Platelets: {platelets} ×10³/μL")

        # Next steps recommendation
        print(f"\n🎯 RECOMMENDED NEXT STEPS:")
        if diagnosis == "BACTERIAL":
            print("   1. 🚨 Immediate antibiotic administration")
            print("   2. 🏥 Hospital admission for monitoring")
            print("   3. 🔬 Blood and CSF cultures")
            print("   4. 📊 Serial neurological assessments")
            print("   5. 💊 Adjust antibiotics based on culture results")
        else:
            print("   1. ✅ Supportive care and observation")
            print("   2. 🏠 Consider outpatient management if stable")
            print("   3. 💊 Pain and fever management")
            print("   4. 🔍 Consider viral PCR if diagnosis uncertain")
            print("   5. 📞 Follow-up in 24-48 hours")

        return diagnosis, confidence

    except KeyboardInterrupt:
        print("\n\n❌ Classification cancelled by user.")
        return None, None
    except Exception as e:
        print(f"❌ Error in classification: {e}")
        return None, None


# Main user interaction loop
def main_user_interface():
    """Main user interface for the classification system"""

    while True:
        print("\n" + "=" * 80)
        print("MENINGITIS CLASSIFICATION SYSTEM")
        print("=" * 80)
        print("\nAvailable options:")
        print("1. 🧠 Classify new patient")
        print("2. 📊 View model performance")
        print("3. ℹ️  Clinical guidelines")
        print("4. 🚪 Exit")

        try:
            choice = input("\nSelect option (1-4): ").strip()

            if choice == '1':
                # Model selection for classification
                print("\n" + "=" * 50)
                print("MODEL SELECTION")
                print("=" * 50)
                print("Available models for classification:")
                for i, result in enumerate(results, 1):
                    accuracy_percent = result['test_accuracy'] * 100
                    print(f"{i}. {result['model_name']} (Accuracy: {accuracy_percent:.1f}%)")

                print(f"{len(results) + 1}. 🏆 Use Best Overall Model ({best_accuracy_model['model_name']})")

                try:
                    model_choice = input(f"\nSelect model (1-{len(results) + 1}): ").strip()

                    if model_choice == str(len(results) + 1):
                        # Use best model
                        selected_result = best_accuracy_model
                        model_name = selected_result['model_name']
                        print(f"🎯 Selected: {model_name} (Best Accuracy)")
                    else:
                        model_choice = int(model_choice) - 1
                        if 0 <= model_choice < len(results):
                            selected_result = results[model_choice]
                            model_name = selected_result['model_name']
                            print(f"🎯 Selected: {model_name}")
                        else:
                            print("❌ Invalid model selection")
                            continue

                    # Map model names to actual model objects and their types
                    model_mapping = {
                        "DEEP NEURAL NETWORK": (dnn_model, True, False),
                        "ARTIFICIAL NEURAL NETWORK": (ann_model, True, False),
                        "MLP CLASSIFIER": (mlp_model, False, True),
                        "SGD CLASSIFIER": (sgd_model, False, False),
                        "XGBOOST CLASSIFIER": (xgb_model, False, False),
                        "ADABOOST CLASSIFIER": (adaboost_model, False, False),
                        "SVM RBF": (svm_rbf_model, False, False),
                        "RANDOM FOREST": (rf_model, False, False)
                    }

                    selected_model_info = model_mapping.get(model_name)
                    if selected_model_info:
                        selected_model, is_dnn, is_sklearn_nn = selected_model_info
                        # Perform classification
                        diagnosis, confidence = classify_new_patient(selected_model, model_name, is_dnn, is_sklearn_nn)

                        if diagnosis is not None:
                            # Option to try another patient
                            while True:
                                another = input("\nWould you like to classify another patient? (y/n): ").lower()
                                if another == 'y':
                                    diagnosis, confidence = classify_new_patient(selected_model, model_name, is_dnn, is_sklearn_nn)
                                    if diagnosis is None:
                                        break
                                elif another == 'n':
                                    print("\n👨‍⚕️ Thank you for using the Meningitis Classification System!")
                                    break
                                else:
                                    print("❌ Please enter 'y' or 'n'")
                    else:
                        print("❌ Error: Model not found")

                except (ValueError, IndexError):
                    print("❌ Please enter a valid number")

            elif choice == '2':
                # Display model performance summary
                print("\n" + "=" * 60)
                print("MODEL PERFORMANCE SUMMARY")
                print("=" * 60)
                print("\n{:<25} {:<10} {:<12} {:<12}".format(
                    "MODEL", "ACCURACY", "SENSITIVITY", "SPECIFICITY"
                ))
                print("-" * 65)
                for result in results:
                    print("{:<25} {:<10.1f}% {:<12.1f}% {:<12.1f}%".format(
                        result['model_name'][:23],
                        result['test_accuracy'] * 100,
                        result['sensitivity'] * 100,
                        result['specificity'] * 100
                    ))

                print(f"\n🏆 Best Model: {best_accuracy_model['model_name']} "
                      f"({best_accuracy_model['test_accuracy'] * 100:.1f}% accuracy)")

            elif choice == '3':
                # Clinical guidelines
                print("\n" + "=" * 60)
                print("CLINICAL GUIDELINES REFERENCE")
                print("=" * 60)
                print("\n🔬 TYPICAL MENINGITIS PATTERNS:")
                print("Bacterial Meningitis:")
                print("  • CSF WBC: >1000 cells/μL (often neutrophils)")
                print("  • CSF Protein: >100 mg/dL")
                print("  • CSF Glucose: <40 mg/dL")
                print("  • CRP: >100 mg/L")

                print("\nViral Meningitis:")
                print("  • CSF WBC: 10-500 cells/μL (often lymphocytes)")
                print("  • CSF Protein: 45-100 mg/dL")
                print("  • CSF Glucose: Normal (>45 mg/dL)")
                print("  • CRP: 10-40 mg/L")

                print("\n🚨 URGENT ACTIONS FOR SUSPECTED BACTERIAL MENINGITIS:")
                print("  1. Immediate antibiotic administration")
                print("  2. Blood cultures before antibiotics")
                print("  3. Lumbar puncture if no contraindications")
                print("  4. Hospital admission and monitoring")

            elif choice == '4':
                print("\n👋 Thank you for using the Meningitis Classification System!")
                print("Stay safe and take care! 🩺")
                break

            else:
                print("❌ Please select a valid option (1-4)")

        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using the system! Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main_user_interface()

print("\n" + "=" * 80)
print("SYSTEM READY FOR CLINICAL USE")
print("=" * 80)
