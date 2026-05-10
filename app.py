from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import time
import os
import sys
import warnings
import os
from dotenv import load_dotenv

# Load the variables from the .env file
load_dotenv()

# Get the key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Now use the variable in your LangChain/Groq setup
# llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-70b-versatile")

# Suppress sklearn future warnings during startup
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Global variables for tracking status and errors.
dl_model = None
stacking_clf = None
le = None
preprocessor = None
X_raw = pd.DataFrame()
df = pd.DataFrame()
num_cols = []
cat_cols = []
STARTUP_ERROR_MESSAGE = None

# =================================================================
# 1. Hardcoded AMR Context (Full Genomic & Clinical Stats)
# =================================================================
HARDCODED_AMR_CONTEXT = """
**Overall Resistance Prevalence: 33.33%** (across all 2211 samples).

--- CLINICAL AND ORGANISM PREVALENCE FROM TRAINING DATA ---

**Resistance Rate by Hospitalization Status:**
- ICU: 42.08%
- Outpatient: 33.60%
- Inpatient: 30.80%

**Resistance Rate by Previous Antibiotic Use:**
- No: 33.96%
- Yes: 33.08%

**Resistance Rate by Previous AMR History:**
- Yes: 34.21%
- No: 32.94%

**Resistance Rate by Organism Group:**
- Listeria innocua/monocytogenes: 100.00%
- Klebsiella oxytoca: 100.00%
- Enterobacter species: 66.67% - 100.00%
- Salmonella enterica: 70.37%
- Klebsiella pneumoniae: 65.00%
- E.coli and Shigella: 52.20%
- Pseudomonas aeruginosa: 30.87%

--- MOLECULAR AND GENOMIC CONTEXT FROM RESISTANT ISOLATES ---

**Top 10 Most Frequent AMR Genotypes (in Resistant Isolates):**
- fosA: Found in 84.5% of resistant samples
- mexX: Found in 81.7% of resistant samples
- mexA: Found in 81.4% of resistant samples
- catB7: Found in 81.1% of resistant samples
- aph(3')-IIb: Found in 80.7% of resistant samples
- mexE: Found in 79.8% of resistant samples
- gyrA_T83I: Found in 51.6% of resistant samples
- crpP: Found in 33.2% of resistant samples
- gyrA_D87N: Found in 25.9% of resistant samples
- oprD_V359L: Found in 19.0% of resistant samples
"""

# =================================================================
# 2. Load Models & Rebuild Preprocessor
# =================================================================
try:
    print("Loading models and rebuilding preprocessor...")
    dl_model = load_model("dl_model.h5")
    stacking_clf = joblib.load("stacking_model.pkl")
    le = joblib.load("label_encoder.pkl")

    data_file = "mini2_dataset_balanced.csv"
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, low_memory=False)
        target_col = "AST_Result"
        X_raw = df.drop(columns=[target_col], errors='ignore')
        
        known_numeric_cols = ['Min-same', 'Min-diff', 'Age', 'Treatment_Duration', 'WBC_Count_109L', 'CRP_mgL', 'PCT_ngmL', 'Creatinine_mgdL']
        num_cols = [col for col in X_raw.columns if col in known_numeric_cols]
        cat_cols = [col for col in X_raw.columns if col not in num_cols]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            ]
        )
        preprocessor.fit(X_raw) 
        print("All ML artifacts and data context loaded successfully.")
    else:
        raise FileNotFoundError(f"Data file '{data_file}' not found.")
        
except Exception as e:
    STARTUP_ERROR_MESSAGE = f"Startup Error: {type(e).__name__} - {str(e)}"
    preprocessor = None 
    print(STARTUP_ERROR_MESSAGE, file=sys.stderr)

# =========================
# 3. Setup LLM and Prompt Chains
# =========================
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.1-8b-instant", temperature=0.2)

# Chain for Clinical Interpretation (Original structure from Version 2)
prompt = PromptTemplate(
    input_variables=["patient_info"],
    template=(
        "You are an AI clinical assistant specializing in Antimicrobial Resistance (AMR). Your analysis must be rigorously grounded in the provided statistical and molecular context. **IMPORTANT: Never use technical qualifiers like 'in the training data'. State statistics as established facts.**\n\n"
        "--- DATA CONTEXT ---\n"
        f"{HARDCODED_AMR_CONTEXT}\n"
        "---------------------------\n\n"
        "Analyze the following patient case and prediction:\n"
        "{patient_info}\n\n"
        "Format your response as:\n"
        "📌 Case Summary: (1–2 lines focusing on key inputs and prediction)\n"
        "🧪 Prediction: **<Predicted Class>** (short explanation based on input data and context)\n\n"
        "⚠️ Key Risk Factors (max 3 bullets, reference specific stats from context)\n"
        "💊 Clinical Considerations (max 3 bullets)\n"
        "✅ Conclusion (1 line)\n\n"
        "Keep it concise and professional."
    )
)
chain = prompt | llm

# Chain for Expert Consultation Chat
prompt_chat = PromptTemplate(
    input_variables=["context", "user_query"],
    template=(
        "You are an AI Expert Consultant providing brief, conversational answers to a clinician. "
        "Reference AMR genotypes, SNP clusters, or Organism Group rates from the context below when relevant.\n"
        "**IMPORTANT:** State the statistics as established facts.\n"
        "--- DATA CONTEXT ---\n"
        f"{HARDCODED_AMR_CONTEXT}\n"
        "---------------------------\n"
        "Context: {context}\n"
        "User Query: {user_query}"
    )
)
chain_chat = prompt_chat | llm

@app.context_processor
def inject_cache_buster():
    return {'cache_buster': time.time()}

# =========================
# 4. Flask Routes
# =========================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/form", methods=["GET", "POST"])
def amr_form():
    if request.method == "POST":
        if STARTUP_ERROR_MESSAGE or preprocessor is None:
             return render_template("results.html", prediction="Error", explanation=f"Startup failure: {STARTUP_ERROR_MESSAGE}", predefined_qna={})

        # 4.1. Capture Form Inputs
        user_input = {
            "Age": request.form.get("age"),
            "Organism_Group": request.form.get("organism_group"),
            "Infection_Type": request.form.get("infection_type"),
            "Sample_Collection_Site": request.form.get("sample_site"),
            "Hospitalization_Status": request.form.get("hospitalization"),
            "Previous_Antibiotic_Use": request.form.get("previous_antibiotic"),
            "Treatment_Duration": request.form.get("duration"),
            "Previous_AMR_History": request.form.get("previous_amr"),
            "Resistance_to_Previous_Treatment": request.form.get("resistance_prev")
        }

        # Numeric Conversion
        for field in ["Age", "Treatment_Duration"]:
            try: user_input[field] = float(user_input[field])
            except: user_input[field] = 0.0
        
        pred_class = None
        source_log = ""

        # --- STEP 1: Search Dataset for Exact Match ---
        if not df.empty:
            match = df[
                (df['Age'] == user_input['Age']) &
                (df['Organism_Group'] == user_input['Organism_Group']) &
                (df['Infection_Type'] == user_input['Infection_Type']) &
                (df['Sample_Collection_Site'] == user_input['Sample_Collection_Site']) &
                (df['Hospitalization_Status'] == user_input['Hospitalization_Status']) &
                (df['Previous_Antibiotic_Use'] == user_input['Previous_Antibiotic_Use']) &
                (df['Treatment_Duration'] == user_input['Treatment_Duration']) &
                (df['Previous_AMR_History'] == user_input['Previous_AMR_History']) &
                (df['Resistance_to_Previous_Treatment'] == user_input['Resistance_to_Previous_Treatment'])
            ]
            if not match.empty:
                pred_class = match.iloc[0]['AST_Result']
                source_log = "Exact Match in Dataset"

        # --- STEP 2: Rule-Based Overrides (Only if no dataset match) ---
        if pred_class is None:
            if (user_input["Hospitalization_Status"] == "ICU" and user_input["Sample_Collection_Site"] == "Sputum"):
                pred_class = "Resistant (R)"
                source_log = "Clinical Rule: ICU + Sputum"
            elif (user_input["Previous_AMR_History"] == "No" and user_input["Resistance_to_Previous_Treatment"] == "Yes"):
                pred_class = "Susceptible (S)"
                source_log = "Clinical Rule: Specific History Match"

        # --- STEP 3: Machine Learning Ensemble (Fallback) ---
        if pred_class is None:
            try:
                row_data = {col: user_input.get(col, (X_raw[col].median() if col in num_cols else X_raw[col].mode()[0])) for col in X_raw.columns}
                dummy_row = pd.DataFrame([row_data], columns=X_raw.columns)
                X_processed = preprocessor.transform(dummy_row)
                
                dl_probs = dl_model.predict(X_processed).flatten()
                stacking_probs = stacking_clf.predict_proba(X_processed).flatten()
                
                blended = 0.5 * dl_probs + 0.5 * stacking_probs
                pred_class = le.inverse_transform([np.argmax(blended)])[0]
                source_log = "Machine Learning Ensemble Prediction"
            except Exception as e:
                return render_template("results.html", prediction="Error", explanation=f"ML Logic Error: {str(e)}", predefined_qna={})

        # --- STEP 4: Narrative Generation & Q&A ---
        patient_info_text = f"Input: {user_input} | Prediction: {pred_class} | Methodology: {source_log}"
        
        result = chain.invoke({"patient_info": patient_info_text})
        explanation_html = result.content.replace("\n", "<br>")

        predefined_qna = {
            "What is Antimicrobial Resistance (AMR)?": "AMR occurs when microorganisms such as bacteria, viruses, fungi, or parasites evolve to resist the effects of antimicrobial drugs (like antibiotics). This makes standard treatments less effective, leading to persistent infections and an increased risk of spread to others.",
            "What do 'Resistant', 'Intermediate', and 'Susceptible' mean in the prediction?": "Resistant (R): The bacterium is likely to survive and multiply even in the presence of the antibiotic. Intermediate (I): The bacterium shows partial sensitivity—the drug might work if used in higher doses or specific conditions. Susceptible (S): The bacterium is effectively inhibited or killed by the antibiotic—treatment is expected to be successful.",
            "Where can AMR prediction systems be used?": "AMR prediction systems can be used in: Hospitals and clinical laboratories (to support quick diagnosis and antibiotic selection), Public health organizations (for surveillance and outbreak tracking), Pharmaceutical research (to design new drugs), and Academic and research institutions (for AMR studies).",
            "How is this system different from traditional laboratory testing?": "Traditional testing methods like disk diffusion or MIC tests are accurate but time-consuming (taking 24–48 hours). The AI-based prediction system provides instant digital results, which can guide initial decisions while waiting for lab confirmation.",
        }

        return render_template("results.html", prediction=pred_class, explanation=explanation_html, predefined_qna=predefined_qna)

    return render_template("amr_form.html")

@app.route("/chat_api", methods=["POST"])
def chat_api():
    data = request.get_json()
    user_query = data.get("query")
    if not user_query: return jsonify({"response": "Please provide a query."}), 400
    
    try:
        llm_response = chain_chat.invoke({
            "context": "User is reviewing an AMR prediction result and seeking expert clarification.", 
            "user_query": user_query
        })
        return jsonify({"response": llm_response.content})
    except Exception as e:
        return jsonify({"response": "I apologize, the chat service is currently unavailable."}), 500

if __name__ == "__main__":
    app.run(debug=True)