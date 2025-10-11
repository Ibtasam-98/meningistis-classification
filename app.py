import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="NeuroScan AI - Meningitis Diagnosis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, professional CSS without gradients
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }

    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }

    .section-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }

    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .success-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .danger-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .stButton>button {
        background: #3498db;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background: #2980b9;
    }

    .feature-input {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
    }

    .model-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e9ecef;
        border-left: 4px solid #3498db;
    }

    .diagnosis-banner {
        background: #2c3e50;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class MeningitisClassifier:
    def __init__(self):
        self.models = {}
        self.results = []
        self.best_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = ['Age', 'WBC_Count', 'Protein_Level', 'Glucose_Level',
                                'Hemoglobin', 'WBC_Blood_Count', 'Platelets', 'CRP_Level']
        self.models_trained = False

    def load_and_prepare_data(self):
        """Load and prepare the meningitis dataset"""
        try:
            # Generate synthetic data for demonstration
            np.random.seed(42)
            n_samples = 1000

            data = {
                'Age': np.random.randint(0, 80, n_samples),
                'WBC_Count': np.random.lognormal(5, 1, n_samples),
                'Protein_Level': np.random.lognormal(4, 0.5, n_samples),
                'Glucose_Level': np.random.normal(60, 20, n_samples),
                'Hemoglobin': np.random.normal(13, 2, n_samples),
                'WBC_Blood_Count': np.random.lognormal(9, 0.5, n_samples),
                'Platelets': np.random.normal(250, 50, n_samples),
                'CRP_Level': np.random.lognormal(3, 1, n_samples),
                'Diagnosis': np.random.choice(['Bacterial', 'Viral'], n_samples, p=[0.4, 0.6])
            }

            df_classification = pd.DataFrame(data)

            # Adjust values based on diagnosis to create realistic patterns
            bacterial_mask = df_classification['Diagnosis'] == 'Bacterial'
            df_classification.loc[bacterial_mask, 'WBC_Count'] *= 2
            df_classification.loc[bacterial_mask, 'Protein_Level'] *= 1.5
            df_classification.loc[bacterial_mask, 'Glucose_Level'] *= 0.7
            df_classification.loc[bacterial_mask, 'CRP_Level'] *= 3

            if df_classification.empty or len(df_classification) < 2:
                st.error("Error: Insufficient data after preprocessing.")
                return None, None

            X = df_classification[self.feature_columns]
            y = df_classification['Diagnosis']

            # Encode target (Bacterial: 1, Viral: 0)
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            return (X_train_scaled, X_test_scaled, y_train, y_test), df_classification

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None

    def create_dnn_model(self, input_shape):
        """Create Deep Neural Network model"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
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

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        return model

    def create_ann_model(self, input_shape):
        """Create Artificial Neural Network model"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def initialize_models(self):
        """Initialize all machine learning models"""
        # Neural Network Models
        self.models['DNN'] = None  # Will be created during training
        self.models['ANN'] = None  # Will be created during training

        # MLP Classifier (Scikit-learn)
        self.models['MLP'] = MLPClassifier(
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

        # Traditional ML Models
        self.models['SGD'] = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.001,
            max_iter=1000,
            random_state=42
        )

        self.models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )

        self.models['AdaBoost'] = AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        )

        self.models['SVM'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )

        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all models"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )

        # Create DNN and ANN models
        dnn_model = self.create_dnn_model(X_train.shape[1])
        ann_model = self.create_ann_model(X_train.shape[1])

        self.models['DNN'] = dnn_model
        self.models['ANN'] = ann_model

        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.empty()

        for i, (name, model) in enumerate(self.models.items()):
            status_text.text(f"Training {name}...")

            start_time = time.time()

            if name in ['DNN', 'ANN']:
                # Keras models training
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                training_time = time.time() - start_time

                y_pred_proba = model.predict(X_test, verbose=0).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)

                test_accuracy = accuracy_score(y_test, y_pred)
                train_accuracy = history.history['accuracy'][-1]
                val_accuracy = history.history['val_accuracy'][-1]

            else:
                # Scikit-learn models training
                model.fit(X_train, y_train)
                training_time = time.time() - start_time

                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model,
                                                                            'predict_proba') else model.decision_function(
                    X_test)

                test_accuracy = accuracy_score(y_test, y_pred)
                train_accuracy = accuracy_score(y_train, model.predict(X_train))
                val_accuracy = train_accuracy

            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)

            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) != 0 else 0.0
                specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0.0
            else:
                sensitivity = test_recall
                specificity = 0.0

            result = {
                'model_name': name,
                'test_accuracy': test_accuracy,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'roc_auc': roc_auc,
                'training_time': training_time,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': test_precision,
                'fpr': fpr,
                'tpr': tpr,
                'cm': cm,
                'model': model
            }

            self.results.append(result)
            progress_bar.progress((i + 1) / len(self.models))

            # Update results in real-time
            with results_container.container():
                st.success(f"{name} trained successfully!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{test_accuracy * 100:.2f}%")
                with col2:
                    st.metric("ROC AUC", f"{roc_auc * 100:.2f}%")
                with col3:
                    st.metric("Training Time", f"{training_time:.2f}s")

        self.best_model = max(self.results, key=lambda x: x['test_accuracy'])
        self.models_trained = True
        status_text.text(" All models trained successfully!")

    def validate_clinical_ranges(self, age, wbc_count, protein_level, glucose_level,
                                 hemoglobin, wbc_blood_count, platelets, crp_level):
        """Validate if inputs are within clinically plausible ranges"""
        warnings = []

        if age < 0 or age > 120:
            warnings.append("Age outside typical range (0-120 years)")
        elif age < 1:
            warnings.append("⚠ Infant patient - special considerations needed")

        if wbc_count < 0:
            warnings.append("CSF WBC count cannot be negative")
        elif wbc_count > 10000:
            warnings.append("⚠ Very high CSF WBC count - check measurement")

        if protein_level < 0:
            warnings.append("CSF Protein level cannot be negative")
        elif protein_level > 500:
            warnings.append("⚠ Very high CSF Protein level - check measurement")

        if glucose_level < 0:
            warnings.append("CSF Glucose level cannot be negative")
        elif glucose_level > 200:
            warnings.append("⚠ High CSF Glucose level - unusual for meningitis")

        if hemoglobin < 0:
            warnings.append("Hemoglobin cannot be negative")
        elif hemoglobin > 20:
            warnings.append("⚠ High Hemoglobin level - check measurement")
        elif hemoglobin < 7:
            warnings.append("⚠ Low Hemoglobin - possible anemia")

        if wbc_blood_count < 0:
            warnings.append("Blood WBC count cannot be negative")
        elif wbc_blood_count > 50000:
            warnings.append("⚠ Very high Blood WBC count - check measurement")

        if platelets < 0:
            warnings.append("Platelet count cannot be negative")
        elif platelets > 1000:
            warnings.append("⚠ High Platelet count - check measurement")
        elif platelets < 50:
            warnings.append("⚠ Low Platelet count - risk of bleeding")

        if crp_level < 0:
            warnings.append("CRP level cannot be negative")
        elif crp_level > 500:
            warnings.append("⚠ Very high CRP level - severe inflammation")

        return warnings

    def classify_patient(self, model_name, patient_data):
        """Classify a new patient using the specified model"""
        try:
            model_result = next((r for r in self.results if r['model_name'] == model_name), None)
            if not model_result:
                st.error(f"Model {model_name} not found")
                return None, None

            model = model_result['model']
            patient_data_scaled = self.scaler.transform([patient_data])

            # Make prediction
            if model_name in ['DNN', 'ANN']:
                prediction_proba = model.predict(patient_data_scaled, verbose=0)[0][0]
                prediction = 1 if prediction_proba > 0.5 else 0
            else:
                prediction_proba = model.predict_proba(patient_data_scaled)[0][1]
                prediction = model.predict(patient_data_scaled)[0]

            diagnosis = "BACTERIAL" if prediction == 1 else "VIRAL"
            confidence = prediction_proba if prediction == 1 else 1 - prediction_proba

            return diagnosis, confidence

        except Exception as e:
            st.error(f"Error in classification: {e}")
            return None, None


def create_performance_plots(classifier):
    """Create interactive performance visualization plots"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Accuracy', 'ROC AUC Scores',
                        'Training Time', 'Clinical Metrics'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    model_names = [r['model_name'] for r in classifier.results]
    test_accuracies = [r['test_accuracy'] * 100 for r in classifier.results]
    roc_aucs = [r['roc_auc'] * 100 for r in classifier.results]
    training_times = [r['training_time'] for r in classifier.results]
    sensitivities = [r['sensitivity'] * 100 for r in classifier.results]
    specificities = [r['specificity'] * 100 for r in classifier.results]

    # Color scheme for 8 models
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#d35400']

    # Accuracy plot
    fig.add_trace(
        go.Bar(name='Test Accuracy', x=model_names, y=test_accuracies,
               marker_color=colors, text=test_accuracies, texttemplate='%{text:.1f}%',
               textposition='auto'),
        row=1, col=1
    )

    # ROC AUC plot
    fig.add_trace(
        go.Bar(name='ROC AUC', x=model_names, y=roc_aucs,
               marker_color=colors, text=roc_aucs, texttemplate='%{text:.1f}%',
               textposition='auto'),
        row=1, col=2
    )

    # Training time plot
    fig.add_trace(
        go.Bar(name='Training Time (s)', x=model_names, y=training_times,
               marker_color=colors, text=training_times, texttemplate='%{text:.2f}s',
               textposition='auto'),
        row=2, col=1
    )

    # Clinical metrics plot
    fig.add_trace(
        go.Bar(name='Sensitivity', x=model_names, y=sensitivities,
               marker_color='#27ae60', text=sensitivities, texttemplate='%{text:.1f}%'),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(name='Specificity', x=model_names, y=specificities,
               marker_color='#c0392b', text=specificities, texttemplate='%{text:.1f}%'),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Model Performance Dashboard",
        template="plotly_white",
    )

    return fig


def create_confidence_gauge(confidence):
    """Create a confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': 'lightcoral'},
                {'range': [60, 80], 'color': 'yellow'},
                {'range': [80, 100], 'color': 'lightgreen'}],
        }))

    fig.update_layout(height=250)
    return fig


def main():
    if 'classifier' not in st.session_state:
        st.session_state.classifier = MeningitisClassifier()
        st.session_state.data_loaded = False

    # Header
    st.markdown('<h1 class="main-header">NeuroScan AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Meningitis Classification System</p>',
                unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Train Models", "Diagnose"])

    with tab1:
        st.markdown('<h2 class="sub-header">System Overview</h2>', unsafe_allow_html=True)

        if not st.session_state.data_loaded:
            with st.spinner("Loading medical dataset..."):
                data, df_classification = st.session_state.classifier.load_and_prepare_data()
                if data is not None:
                    st.session_state.data = data
                    st.session_state.df_classification = df_classification
                    st.session_state.data_loaded = True
                    st.success("Medical dataset loaded successfully!")

        if st.session_state.data_loaded:
            df_classification = st.session_state.df_classification

            st.subheader("Dataset Analytics")
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("Total Patients", len(df_classification))
            with info_col2:
                bacterial_count = len(df_classification[df_classification['Diagnosis'] == 'Bacterial'])
                st.metric("Bacterial Cases", bacterial_count)
            with info_col3:
                viral_count = len(df_classification[df_classification['Diagnosis'] == 'Viral'])
                st.metric("Viral Cases", viral_count)

            st.subheader("Disease Distribution")
            class_dist = df_classification['Diagnosis'].value_counts()
            fig_pie = px.pie(
                values=class_dist.values,
                names=class_dist.index,
                color=class_dist.index,
                color_discrete_map={'Bacterial': '#e74c3c', 'Viral': '#3498db'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.markdown('<h2 class="sub-header">Model Training</h2>', unsafe_allow_html=True)

        if st.button("Train AI Models", use_container_width=True):
            if st.session_state.data_loaded:
                with st.spinner("Training AI models..."):
                    st.session_state.classifier.initialize_models()
                    st.session_state.classifier.train_and_evaluate_models(*st.session_state.data)
                st.success("AI models trained successfully!")
            else:
                st.error("Please load data first!")

        if st.session_state.classifier.models_trained:
            st.subheader("Model Comparison")
            metrics_data = []
            for result in st.session_state.classifier.results:
                metrics_data.append({
                    'Model': result['model_name'],
                    'Accuracy': f"{result['test_accuracy'] * 100:.2f}%",
                    'ROC AUC': f"{result['roc_auc'] * 100:.2f}%",
                    'Sensitivity': f"{result['sensitivity'] * 100:.2f}%",
                    'Specificity': f"{result['specificity'] * 100:.2f}%",
                    'Training Time': f"{result['training_time']:.2f}s"
                })

            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)

            # Display best model info
            best_model = st.session_state.classifier.best_model
            st.subheader("🏆 Best Performing Model")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model", best_model['model_name'])
            with col2:
                st.metric("Accuracy", f"{best_model['test_accuracy'] * 100:.2f}%")
            with col3:
                st.metric("ROC AUC", f"{best_model['roc_auc'] * 100:.2f}%")
            with col4:
                st.metric("Training Time", f"{best_model['training_time']:.2f}s")

            st.subheader("Performance Analytics")
            fig_performance = create_performance_plots(st.session_state.classifier)
            st.plotly_chart(fig_performance, use_container_width=True)

    with tab3:
        st.markdown('<h2 class="sub-header">Patient Diagnosis</h2>', unsafe_allow_html=True)

        if not st.session_state.classifier.models_trained:
            st.warning("Please train AI models first using the 'Train AI Models' button.")
        else:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("AI Model Selection")
                model_names = [r['model_name'] for r in st.session_state.classifier.results]
                selected_model = st.selectbox(
                    "Choose AI Model:",
                    model_names,
                    index=model_names.index(st.session_state.classifier.best_model['model_name'])
                )

                st.subheader("Clinical Parameters")
                with st.form("patient_data_form"):
                    age = st.slider("Age (years)", 0, 120, 35)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        wbc_count = st.number_input("WBC Count (cells/μL)", min_value=0.0, value=150.0, step=10.0)
                        protein_level = st.number_input("Protein (mg/dL)", min_value=0.0, value=80.0, step=5.0)
                    with col_b:
                        glucose_level = st.number_input("Glucose (mg/dL)", min_value=0.0, value=60.0, step=5.0)

                    col_c, col_d = st.columns(2)
                    with col_c:
                        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, value=13.5, step=0.5)
                        wbc_blood_count = st.number_input("Blood WBC (cells/μL)", min_value=0.0, value=12000.0,
                                                          step=1000.0)
                    with col_d:
                        platelets = st.number_input("Platelets (×10³/μL)", min_value=0.0, value=250.0, step=50.0)
                        crp_level = st.number_input("CRP (mg/L)", min_value=0.0, value=45.0, step=5.0)

                    submitted = st.form_submit_button("Run AI Diagnosis", use_container_width=True)

            with col2:
                if submitted:
                    warnings = st.session_state.classifier.validate_clinical_ranges(
                        age, wbc_count, protein_level, glucose_level,
                        hemoglobin, wbc_blood_count, platelets, crp_level
                    )

                    if warnings:
                        for warning in warnings:
                            st.warning(warning)

                    patient_data = [age, wbc_count, protein_level, glucose_level,
                                    hemoglobin, wbc_blood_count, platelets, crp_level]

                    with st.spinner("AI Analysis in Progress..."):
                        diagnosis, confidence = st.session_state.classifier.classify_patient(selected_model,
                                                                                             patient_data)

                    if diagnosis:
                        st.markdown("""
                        <div class='diagnosis-banner'>
                            <h2>AI Diagnosis Complete</h2>
                        </div>
                        """, unsafe_allow_html=True)

                        if diagnosis == "BACTERIAL":
                            st.markdown("""
                            <div class='danger-box'>
                                <h2>BACTERIAL MENINGITIS</h2>
                                <p>High-Risk Condition - Immediate Intervention Required</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class='success-box'>
                                <h2>VIRAL MENINGITIS</h2>
                                <p>Lower-Risk Condition - Supportive Management</p>
                            </div>
                            """, unsafe_allow_html=True)

                        fig_gauge = create_confidence_gauge(confidence)
                        st.plotly_chart(fig_gauge, use_container_width=True)

                        st.subheader("Clinical Recommendations")
                        if diagnosis == "BACTERIAL":
                            st.markdown("""
                            **Immediate Actions Required:**
                            - Empiric antibiotics within 1 hour
                            - Blood cultures before antibiotics
                            - ICU admission preparation
                            - Ceftriaxone 2g IV + Vancomycin
                            - Dexamethasone if indicated
                            - Close neurological monitoring
                            """)
                        else:
                            st.markdown("""
                            **Supportive Management:**
                            - Supportive care and observation
                            - Outpatient management if stable
                            - Pain and fever management
                            - Adequate hydration
                            - Follow-up in 24-48 hours
                            - Consider viral PCR testing
                            """)

                        # Show model used
                        st.info(f"**Model Used:** {selected_model} | **Confidence:** {confidence * 100:.2f}%")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>NeuroScan AI - Meningitis Classification System</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
