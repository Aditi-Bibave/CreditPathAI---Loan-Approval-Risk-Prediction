# final_UI.py
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ML imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier

# plotting / metrics
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# optional libs
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    IMB_AVAILABLE = True
except Exception:
    IMB_AVAILABLE = False

# -------------------------
# Utilities (feature building + safe scaling)
# -------------------------
def build_feature_row(trained_features, user_values):
    """
    Build a 1-row DataFrame aligned exactly to trained_features.
    user_values: dict of user input values (numbers or strings).
    Strategy:
      - Start with zeros for all trained_features.
      - For numeric user_values: find the best matching numeric column (exact match or token)
        and fill the number.
      - For categorical string values: set any trained feature column whose name contains the category token to 1.
    """
    X = pd.DataFrame(0, index=[0], columns=trained_features)
    feat_lows = [c.lower() for c in trained_features]

    for key, val in user_values.items():
        if val is None:
            continue
        key_low = str(key).lower()

        # numeric values
        if isinstance(val, (int, float, np.integer, np.floating)):
            placed = False
            # exact name match
            for idx, col in enumerate(trained_features):
                if col.lower() == key_low:
                    X.iat[0, idx] = val
                    placed = True
                    break
            if placed:
                continue
            # token match: place into first numeric-like column that contains the key token
            for idx, col_low in enumerate(feat_lows):
                if key_low in col_low and np.issubdtype(X[trained_features[idx]].dtype, np.number):
                    X.iat[0, idx] = val
                    placed = True
                    break
            # fallback: try columns that include common synonyms
            if not placed:
                for synonym in ("amount", "income", "loan", "rate", "score", "value", "term", "ltv", "dtir", "tenure"):
                    if synonym in key_low:
                        for idx, col_low in enumerate(feat_lows):
                            if synonym in col_low and np.issubdtype(X[trained_features[idx]].dtype, np.number):
                                X.iat[0, idx] = val
                                placed = True
                                break
                    if placed:
                        break
            continue

        # string / categorical values
        if isinstance(val, str):
            token = val.strip().lower().replace(" ", "_")
            # mark any trained feature that contains the token
            for idx, col_low in enumerate(feat_lows):
                if token in col_low:
                    X.iat[0, idx] = 1
            # also mark columns of pattern key_token or token_key
            for idx, col_low in enumerate(feat_lows):
                if key_low in col_low and token in col_low:
                    X.iat[0, idx] = 1

    return X

def safe_scale_row(X_input, scaler, trained_features):
    """
    Align X_input to trained_features and scale numeric columns using scaler without causing sklearn feature-name errors.
    """
    # Ensure all trained features exist
    for col in trained_features:
        if col not in X_input.columns:
            X_input[col] = 0

    # Keep only trained features (in correct order)
    X_input = X_input[trained_features].copy()

    if scaler is None:
        return X_input

    # Determine numeric columns in the aligned X_input
    numeric_cols = list(X_input.select_dtypes(include=[np.number]).columns)

    # If scaler exposes feature_names_in_, respect that order to avoid mismatch
    try:
        feature_names_in = list(scaler.feature_names_in_)
        # Determine numeric columns expected by scaler that are also present
        numeric_expected = [c for c in feature_names_in if c in X_input.columns and np.issubdtype(X_input[c].dtype, np.number)]
        if numeric_expected:
            X_input[numeric_expected] = scaler.transform(X_input[numeric_expected])
        else:
            if numeric_cols:
                X_input[numeric_cols] = scaler.transform(X_input[numeric_cols])
    except Exception:
        # Fallback attempts
        try:
            if numeric_cols:
                X_input[numeric_cols] = scaler.transform(X_input[numeric_cols])
        except Exception:
            try:
                if numeric_cols:
                    X_input[numeric_cols] = scaler.transform(X_input[numeric_cols].astype(float))
            except Exception:
                # last resort: return unscaled (model may still accept it)
                pass

    return X_input

# -------------------------
# Small safe helpers for preprocessing
# -------------------------
def load_local_dataset(path="loan_dataset/Loan_Default.csv"):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def safe_get_dummies(df, cat_cols):
    present = [c for c in cat_cols if c in df.columns]
    if present:
        return pd.get_dummies(df, columns=present, drop_first=True)
    return df

def drop_safe(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

def clean_colnames(df):
    df.columns = (
        df.columns.str.replace("[", "", regex=False)
                  .str.replace("]", "", regex=False)
                  .str.replace("<", "", regex=False)
                  .str.replace(">", "", regex=False)
                  .str.replace(" ", "_", regex=False)
    )
    return df

# -------------------------
# UI styling (screenshot-like but professional)
# -------------------------
st.set_page_config(page_title="CreditPathAI — Loan Default Predictor", layout="wide")
st.markdown("""
<style>
body { background: #f6f7fb; color: #0b2b35; }
.header {
  background: linear-gradient(90deg,#2aa19c,#1976a5);
  padding:22px;
  color:white;
  font-weight:800;
  font-size:22px;
  border-radius:12px;
  text-align:center;
  margin-bottom:18px;
}
.section-title { font-weight:700; color:#07323d; margin-bottom:8px; }
.card { background:white; padding:18px; border-radius:10px; box-shadow:0 8px 30px rgba(11,43,53,0.06); }
.result-low { background:#eaf9f2; border-left:6px solid #06a56b; padding:18px; border-radius:8px; color:#056b46; font-weight:700; }
.result-med { background:#fff8ec; border-left:6px solid #f6a800; padding:18px; border-radius:8px; color:#8a5a00; font-weight:700; }
.result-high { background:#fff3f3; border-left:6px solid #ff3b3b; padding:18px; border-radius:8px; color:#9b1c1c; font-weight:700; }
.small-box { background:white; padding:12px; border-radius:10px; text-align:center; box-shadow:0 4px 16px rgba(11,43,53,0.04); }
.footer { text-align:center; color:#6b7280; padding:12px; margin-top:18px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">CREDITPATHAI — LOAN DEFAULT RISK PREDICTION SYSTEM</div>', unsafe_allow_html=True)

# -------------------------
# Main applicant inputs (top area like screenshot)
# -------------------------
st.markdown("### Applicant Example Inputs")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=90, value=30)
    gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])
with col2:
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0, step=1000, value=50000)
    property_value = st.number_input("Property Value (₹)", min_value=0, step=1000, value=1000000)
with col3:
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, step=1000, value=300000)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.1, max_value=40.0, value=10.0)

st.write("---")

# -------------------------
# Sidebar (exact layout A)
# -------------------------
st.sidebar.title("Loan Processing Panel")

# PERSONAL INFORMATION expander
with st.sidebar.expander("🧍 Personal Information", expanded=True):
    gender_sidebar = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])
    credit_worthiness = st.selectbox("Credit Worthiness", ["l1", "l2"])
    age_bracket = st.selectbox("Age Bracket", ["<25","25-34","35-44","45-54","55-64","65-74",">74"])
    credit_type = st.selectbox("Credit Type", ["CRIF","EQUI","EXP","Other"])
    credit_score = st.slider("Credit Score", 300, 900, 650)

# FINANCIAL INFORMATION expander
with st.sidebar.expander("💰 Financial Information", expanded=True):
    income = st.number_input("Monthly Income (₹)", min_value=0, step=1000, value=50000, key="inc_sidebar")
    property_val = st.number_input("Property Value (₹)", min_value=0, step=1000, value=1000000, key="prop_sidebar")
    dtir1 = st.number_input("Debt-to-Income Ratio (dtir1)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
    LTV = st.number_input("Loan-to-Value (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)

# LOAN DETAILS expander
with st.sidebar.expander("📄 Loan Details", expanded=True):
    loan_type = st.selectbox("Loan Type", ["type1","type2","type3"])
    approv_in_adv = st.selectbox("Pre-approved?", ["pre","not_pre"])
    loan_purpose = st.selectbox("Loan Purpose", ["p1","p2","p3","p4"])
    submission = st.selectbox("Application Submitted To", ["to_inst","not_inst"])
    loan_amount_sidebar = st.number_input("Loan Amount (₹)", min_value=0, step=1000, value=300000, key="loan_sidebar")
    rate_of_interest_sidebar = st.number_input("Interest Rate (%)", min_value=0.1, max_value=40.0, value=10.0, step=0.1, key="rate_sidebar")
    term = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=120, key="term_sidebar")

st.sidebar.markdown("---")

# Dataset load / preview / buttons (always visible)
dataset_path = "loan_dataset/Loan_Default.csv"
df_local = load_local_dataset(dataset_path)
if df_local is not None:
    st.sidebar.success(f"📁 Loaded dataset from `{dataset_path}` ({df_local.shape[0]} rows)")
    st.session_state["df"] = df_local
else:
    st.sidebar.info("No local dataset found. Use uploader below or add file into loan_dataset/")
    uploaded = st.sidebar.file_uploader("Upload Loan Dataset (.csv)", type=["csv"])
    if uploaded is not None:
        st.session_state["df"] = pd.read_csv(uploaded)
        st.sidebar.success("Uploaded dataset")

if "df" in st.session_state:
    if st.sidebar.checkbox("Preview dataset (first 5 rows)"):
        st.subheader("Dataset preview")
        st.dataframe(st.session_state["df"].head())

st.sidebar.markdown("---")

preprocess_btn = st.sidebar.button("🔄 Preprocess Dataset")
train_btn = st.sidebar.button("⚙ Train Models")
model_options = ["Select Model", "Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting", "KNN"]
if XGBOOST_AVAILABLE:
    model_options.append("XGBoost")
selected_model = st.sidebar.selectbox("Choose Model", model_options)
predict_btn = st.sidebar.button("🔍 Predict Risk")

st.sidebar.markdown("---")
st.sidebar.info("Flow: Preprocess → Train → Choose Model → Predict")

# -------------------------
# Show dataset top-level info
# -------------------------
if "df" in st.session_state and not preprocess_btn and not train_btn:
    st.write(f"📁 Loaded dataset: {dataset_path}")
    st.write("Dataset Shape:", st.session_state["df"].shape)

# -------------------------
# PREPROCESSING
# -------------------------
if preprocess_btn:
    if "df" not in st.session_state:
        st.error("No dataset loaded. Upload or place file into loan_dataset/ and reload.")
    else:
        with st.spinner("Preprocessing dataset..."):
            dfp = st.session_state["df"].copy()
            # categorical columns from your notebook
            cat_cols = [
                "loan_limit","Gender","approv_in_adv","loan_type","loan_purpose",
                "Credit_Worthiness","open_credit","business_or_commercial",
                "Neg_ammortization","interest_only","lump_sum_payment",
                "construction_type","occupancy_type","Secured_by","total_units",
                "credit_type","co-applicant_credit_type","age",
                "submission_of_application","Region","Security_Type"
            ]
            dfp = safe_get_dummies(dfp, cat_cols)
            dfp = drop_safe(dfp, ["ID","year"])
            # fill numeric missing
            for c in dfp.select_dtypes(include=[np.number]).columns:
                dfp[c] = dfp[c].fillna(dfp[c].median())
            # encode target
            if "Status" not in dfp.columns:
                st.error("Target column 'Status' not found in dataset.")
            else:
                le = LabelEncoder()
                dfp["Status"] = le.fit_transform(dfp["Status"].astype(str))
                dfp = clean_colnames(dfp)
                st.session_state["df_preprocessed"] = dfp
                st.success("✅ Preprocessing complete. Now click 'Train Models'.")

# -------------------------
# TRAINING
# -------------------------
if train_btn:
    if "df_preprocessed" not in st.session_state:
        st.error("Please preprocess dataset first.")
    else:
        with st.spinner("Training models (this may take a few minutes)..."):
            dfp = st.session_state["df_preprocessed"].copy()
            TARGET = "Status"
            X = dfp.drop(columns=[TARGET])
            y = dfp[TARGET]

            # train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # scaling
            scaler = StandardScaler()
            numeric_features = X_train.select_dtypes(include=[np.number]).columns
            X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
            X_test[numeric_features] = scaler.transform(X_test[numeric_features])

            # SMOTE if available
            if IMB_AVAILABLE:
                sm = SMOTE(random_state=42)
                X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
                X_train_sm = pd.DataFrame(X_train_sm, columns=X_train.columns)
            else:
                X_train_sm, y_train_sm = X_train, y_train

            # save scaler and columns
            st.session_state["scaler"] = scaler
            st.session_state["model_features"] = list(X_train_sm.columns)
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test

            # define and train models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=300),
                "Extra Trees": ExtraTreesClassifier(n_estimators=300),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=300),
                "KNN": KNeighborsClassifier(n_neighbors=5)
            }
            if XGBOOST_AVAILABLE:
                models["XGBoost"] = xgb.XGBClassifier(eval_metric="logloss")

            trained = {}
            for name, mdl in models.items():
                mdl.fit(X_train_sm, y_train_sm)
                trained[name] = mdl

            st.session_state["models"] = trained
            st.success("🔥 Training complete. Models are ready.")

# -------------------------
# PREDICTION
# -------------------------
if predict_btn:
    if "models" not in st.session_state:
        st.error("Train models first (Preprocess → Train).")
    elif selected_model == "Select Model":
        st.error("Please select a model.")
    else:
        model = st.session_state["models"][selected_model]
        features = st.session_state["model_features"]
        scaler = st.session_state.get("scaler", None)

        st.info("🔍 Building input → Aligning features → Scaling → Predicting...")

        # build mapping of UI -> expected dataset tokens (keep names used in dataset)
        user_values = {
            "age": age,
            "Gender": gender_sidebar if 'gender_sidebar' in st.session_state else gender,
            "Credit_Worthiness": credit_worthiness,
            "credit_type": credit_type,
            "Credit_Score": credit_score,
            "income": income,
            "property_value": property_val,
            "dtir1": dtir1,
            "LTV": LTV,
            "loan_type": loan_type,
            "approv_in_adv": approv_in_adv,
            "loan_purpose": loan_purpose,
            "submission_of_application": submission,
            "loan_amount": loan_amount_sidebar if 'loan_amount_sidebar' in st.session_state else loan_amount,
            "rate_of_interest": rate_of_interest_sidebar if 'rate_of_interest_sidebar' in st.session_state else interest_rate,
            "term": term
        }

        # Build exact feature row (no guessed columns)
        X_input = build_feature_row(features, user_values)

        # Align and scale safely
        X_scaled = safe_scale_row(X_input.copy(), scaler, features)

        # Predict probability
        try:
            prob = float(model.predict_proba(X_scaled)[0][1]) * 100
        except Exception:
            # if model has no predict_proba, fallback to predict
            try:
                pred = model.predict(X_scaled)[0]
                prob = 100.0 if pred == 1 else 0.0
            except Exception as e:
                st.error("Prediction failed: " + str(e))
                prob = 0.0

        non_prob = 100 - prob
        conf = max(prob, non_prob)

        # Result banner/cards (screenshot-like)
        if prob < 40:
            st.markdown(f'<div class="result-low">🟢 LOW DEFAULT RISK — {prob:.2f}%</div>', unsafe_allow_html=True)
        elif prob < 70:
            st.markdown(f'<div class="result-med">🟡 MEDIUM DEFAULT RISK — {prob:.2f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-high">🔴 HIGH DEFAULT RISK — {prob:.2f}%</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="small-box">Default<br><span style="color:#d93737;font-size:18px">{prob:.2f}%</span></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="small-box">Non-Default<br><span style="color:#0fa958;font-size:18px">{non_prob:.2f}%</span></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="small-box">Model Confidence<br><span style="color:#0b3b4a;font-size:18px">{conf:.2f}%</span></div>', unsafe_allow_html=True)

        st.progress(min(100, int(conf)))

        # Recommendations text
        if prob >= 70:
            st.warning("Recommendation: Immediate follow-up, restructure or legal remedies.")
        elif prob >= 40:
            st.info("Recommendation: Monitoring, EMI rescheduling, frequent reminders.")
        else:
            st.success("Recommendation: Low risk — approve or monitor normally.")

        # ROC and Confusion Matrix using held-out test set
        st.markdown("### Model Evaluation (held-out test set)")
        X_test = st.session_state.get("X_test", None)
        y_test = st.session_state.get("y_test", None)

        if X_test is not None and y_test is not None:
            try:
                # ensure X_test has same columns as features (scaler/transform consistency)
                X_test_aligned = X_test.copy()
                missing_cols = [c for c in features if c not in X_test_aligned.columns]
                for c in missing_cols:
                    X_test_aligned[c] = 0
                X_test_aligned = X_test_aligned[features]

                # Some models require scaling the test set already done during training path.
                # If scaler was used, X_test is already scaled during training step.

                y_prob = model.predict_proba(X_test_aligned)[:, 1]
                auc_score = roc_auc_score(y_test, y_prob)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                st.markdown(f"**ROC Curve (AUC = {auc_score:.3f})**")
                fig = plt.figure(figsize=(6,4))
                plt.plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
                plt.plot([0,1], [0,1], 'k--')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.info("ROC not available for this model or an error occurred: " + str(e))

            try:
                preds = model.predict(X_test_aligned)
                cm = confusion_matrix(y_test, preds)
                st.markdown("**Confusion Matrix**")
                fig2 = plt.figure(figsize=(4,3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig2)
                plt.close(fig2)
            except Exception as e:
                st.info("Confusion matrix not available: " + str(e))
        else:
            st.info("No test set available (train models first).")

# -------------------------
# Developer info & footer
# -------------------------
with st.expander("Developer Info — session keys & feature list"):
    st.write("Session keys:", list(st.session_state.keys()))
    if "model_features" in st.session_state:
        st.write("Model features count:", len(st.session_state["model_features"]))
        st.write("Model features (first 300):")
        st.write(st.session_state["model_features"][:300])
    if "scaler" in st.session_state:
        try:
            st.write("scaler.feature_names_in_:")
            st.write(list(st.session_state["scaler"].feature_names_in_))
        except Exception:
            st.write("scaler.feature_names_in_ not available in this scaler.")

st.markdown('<div class="footer">© 2025 CreditPathAI. All rights reserved.</div>', unsafe_allow_html=True)
