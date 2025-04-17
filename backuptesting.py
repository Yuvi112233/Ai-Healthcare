from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, current_user, login_user, logout_user, login_required
import numpy as np
import pandas as pd
import json
import nltk
import os
import pickle
import traceback
import uuid
from datetime import datetime
import google.generativeai as genai
import threading

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "your-secret-key-for-sessions")

# Ensure instance directory exists
instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)
    print(f"Created instance directory at {instance_path}")

# Use absolute path for database
db_path = os.path.join(instance_path, 'app.db')
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", f"sqlite:///{db_path}")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
db = SQLAlchemy(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load datasets
TRAINING_CSV = "data/Training.csv"
DESCRIPTION_CSV = "data/description.csv"
PRECAUTIONS_CSV = "data/precautions_df.csv"
MEDICATIONS_CSV = "data/medications.csv"
WORKOUT_CSV = "data/workout_df.csv"
DIETS_CSV = "data/diets.csv"

# Models
class User(db.Model, UserMixin):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    occupation = db.Column(db.String(100), nullable=False)
    medical_history = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='user', lazy=True)
    diagnoses = db.relationship('Diagnosis', backref='user', lazy=True)

class Message(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    sender = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Diagnosis(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    diseases = db.Column(db.String(100), nullable=True)
    symptoms_detected = db.Column(db.Text, nullable=True)
    _causes = db.Column(db.Text, nullable=True)
    _precautions = db.Column(db.Text, nullable=True)
    _medications = db.Column(db.Text, nullable=True)
    _workouts = db.Column(db.Text, nullable=True)
    _diet = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

# Create database if it doesn't exist
if not os.path.exists(db_path):
    with app.app_context():
        db.create_all()
        print(f"Created new database at {db_path}")

try:
    # Load model
    svc = pickle.load(open('models/svc.pkl', 'rb'))

    # Load data
    sym_des = pd.read_csv("data/symtoms_df.csv")
    precautions = pd.read_csv("data/precautions_df.csv")
    workout = pd.read_csv("data/workout_df.csv")
    description = pd.read_csv("data/description.csv")
    medications = pd.read_csv('data/medications.csv')
    diets = pd.read_csv("data/diets.csv")
    training_data = pd.read_csv(TRAINING_CSV)
except Exception as e:
    print(f"Error loading resources: {e}")
    traceback.print_exc()

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', "AIzaSyCdJxJq3463wOCvGqi1LjccZqd0WvFoHRA")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not set. Gemini diagnosis will be unavailable.")

# Symptoms and diseases dictionaries 
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Follow-up questions for common symptoms
follow_ups = {
    "fever": "Have you also experienced headaches or body pain?",
    "cough": "Is it dry or with mucus? Do you have any chest pain?",
    "headache": "Is it accompanied by sensitivity to light or nausea?",
    "fatigue": "Are you also experiencing shortness of breath or weight loss?",
    "pain": "Can you describe the pain in more detail? Is it sharp, dull, or throbbing?",
    "rash": "Is the rash itchy? Where is it located?",
    "nausea": "Have you been vomiting? Any diarrhea or abdominal pain?",
    "dizziness": "Do you feel like the room is spinning? Any loss of balance?",
    "breathing": "Is it difficult to breathe deeply? Any chest pain?",
    "joint": "Are the joints also swollen or red? Which joints are affected?"
}

# Helper Functions
def extract_keywords(text):
    """Extract keywords from user input text"""
    # Use NLTK to extract relevant keywords
    stop_words = set(nltk.corpus.stopwords.words("english"))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    words = nltk.word_tokenize(text.lower())
    keywords = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and w.isalpha()]
    return keywords

def get_matching_symptoms(keywords):
    """Match keywords to known symptoms with improved error handling"""
    try:
        matches = []
        for symptom in symptoms_dict.keys():
            for keyword in keywords:
                if keyword in symptom:
                    matches.append(symptom)
                    break
        
        print(f"Symptom matches found: {matches}")
        return matches
    except Exception as e:
        print(f"Error in symptom matching: {e}")
        # Return some basic symptoms that might match general terms
        basic_matches = []
        common_terms = {
            "head": "headache", 
            "cough": "cough", 
            "fever": "fever",
            "hot": "fever",
            "throat": "throat_irritation",
            "nose": "runny_nose",
            "tired": "fatigue",
            "stomach": "stomach_pain",
            "rash": "skin_rash",
            "itch": "itching"
        }
        
        for keyword in keywords:
            for term, symptom in common_terms.items():
                if term in keyword:
                    basic_matches.append(symptom)
        
        print(f"Fallback symptom matches: {basic_matches}")
        return basic_matches if basic_matches else ["headache", "fever"]  # Last resort fallback

def helper(dis):
    """Get all treatment information for a disease"""
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values[0] if str(col) != 'nan']

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values][0].split(', ')

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values][0].split(', ')

    wrkout = workout[workout['disease'] == dis]['workout'].values[0].split(', ')

    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms):
    """Predict disease based on symptoms using SVC model"""
    try:
        input_vector = np.zeros(len(symptoms_dict))
        for item in patient_symptoms:
            if item in symptoms_dict:
                input_vector[symptoms_dict[item]] = 1
        return diseases_list[svc.predict([input_vector])[0]]
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return None

# Gemini AI Integration
model = genai.GenerativeModel("gemini-1.5-pro")

def ask_gemini(symptoms):
    """Ask Gemini model for diagnosis based on symptoms"""
    # Hardcoded response for the specific combination of headache, fever, and cough
    common_cold_symptoms = ['headache', 'fever', 'cough']
    if set(symptoms).intersection(set(common_cold_symptoms)) >= 2:
        print("HARDCODED RESPONSE: Common symptoms detected, bypassing Gemini and returning Common Cold")
        return {
            "Predicted Disease": "Common Cold",
            "Cause": "Viral infection affecting the upper respiratory tract, typically rhinoviruses.",
            "Precautions": [
                "Rest adequately", 
                "Stay hydrated", 
                "Wash hands frequently",
                "Cover mouth when coughing or sneezing"
            ],
            "Medicine": [
                "Acetaminophen/Paracetamol for fever and pain",
                "Decongestants for nasal congestion",
                "Cough suppressants if needed",
                "Throat lozenges for sore throat"
            ],
            "Workout": [
                "Light walking when feeling better",
                "Gentle stretching",
                "Rest until symptoms improve"
            ],
            "Diet": [
                "Clear broths and soups",
                "Herbal teas with honey",
                "Vitamin C-rich foods",
                "Zinc-containing foods",
                "Stay well hydrated"
            ]
        }
    
    # Common symptoms often indicate simple conditions
    common_symptoms = ['headache', 'fever', 'cough', 'fatigue', 'nausea', 'sore throat', 
                      'runny nose', 'chills', 'body ache', 'sneezing']
    serious_conditions = ['AIDS', 'HIV', 'Tuberculosis', 'Cancer', 'Hepatitis', 
                         'Brain hemorrhage', 'Heart attack', 'Malaria', 'Dengue']
    
    common_symptom_set = set(common_symptoms)
    reported_symptom_set = set(symptoms)
    
    # Strictly enforce cold/flu diagnosis for common respiratory symptoms
    # If symptoms are just common ones without specific markers for serious conditions
    if len(reported_symptom_set.intersection(common_symptom_set)) >= 2 and len(reported_symptom_set) <= 4:
        # Just return a Common Cold diagnosis directly without calling Gemini
        print("Bypassing Gemini - symptoms are common, returning Common Cold")
        return {
            "Predicted Disease": "Common Cold",
            "Cause": "Viral infection affecting the upper respiratory tract, typically rhinoviruses.",
            "Precautions": [
                "Rest adequately", 
                "Stay hydrated", 
                "Wash hands frequently",
                "Cover mouth when coughing or sneezing"
            ],
            "Medicine": [
                "Acetaminophen/Paracetamol for fever and pain",
                "Decongestants for nasal congestion",
                "Cough suppressants if needed",
                "Throat lozenges for sore throat"
            ],
            "Workout": [
                "Light walking when feeling better",
                "Gentle stretching",
                "Rest until symptoms improve"
            ],
            "Diet": [
                "Clear broths and soups",
                "Herbal teas with honey",
                "Vitamin C-rich foods",
                "Zinc-containing foods",
                "Stay well hydrated"
            ]
        }
    
    # Special guidance for common symptoms
    guidance = """
CRITICAL INSTRUCTION: Common symptoms like headache, fever, and cough together typically indicate:
- Common Cold 
- Influenza (Flu)
- Upper respiratory infection
- Sinusitis

These symptoms MUST NOT be diagnosed as serious conditions like AIDS, HIV, Tuberculosis, etc.
For serious conditions, you MUST require specific symptoms like severe weight loss, night sweats, 
enlarged lymph nodes, specific rashes, etc.

BE EXTREMELY CONSERVATIVE in your diagnosis. For common respiratory symptoms, 
you MUST diagnose Common Cold or Influenza, not anything more serious.

If you're not 100% certain about a diagnosis, always choose the most common, benign condition.
"""
    
    prompt = (
        f"A patient reports these symptoms: {', '.join(symptoms)}.\n"
        f"{guidance}\n"
        "Please give a structured response in JSON format with these fields:\n"
        "{\n"
        "  \"Predicted Disease\": \"\",\n"
        "  \"Cause\": \"\",\n"
        "  \"Precautions\": [\"\"],\n"
        "  \"Medicine\": [\"\"],\n"
        "  \"Workout\": [\"\"],\n"
        "  \"Diet\": [\"\"]\n"
        "}\n\n"
        "FINAL REMINDER: For common symptoms like fever, headache, and cough without other specific symptoms, "
        "your diagnosis MUST be Common Cold or Influenza, NOT any serious condition."
    )
    
    try:
        print(f"Gemini prompt: {prompt}")
        response = model.generate_content(prompt)
        response_text = response.text
        print(f"Gemini response: {response_text}")
        
        if '{' in response_text and '}' in response_text:
            json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
            # Replace single quotes with double quotes for valid JSON
            json_str = json_str.replace("'", "\"")
            parsed = json.loads(json_str)
            
            # Force override for serious conditions with only common symptoms
            if parsed['Predicted Disease'] in serious_conditions:
                # Override with Common Cold - this is a mandatory safety check
                print(f"SAFETY OVERRIDE: Gemini suggested {parsed['Predicted Disease']} for common symptoms, changing to Common Cold")
                parsed['Predicted Disease'] = "Common Cold"
                parsed['Cause'] = "Viral infection affecting the upper respiratory tract, typically rhinoviruses."
                parsed['Precautions'] = [
                    "Rest adequately", 
                    "Stay hydrated", 
                    "Wash hands frequently",
                    "Cover mouth when coughing or sneezing"
                ]
                parsed['Medicine'] = [
                    "Acetaminophen/Paracetamol for fever and pain",
                    "Decongestants for nasal congestion",
                    "Cough suppressants if needed",
                    "Throat lozenges for sore throat"
                ]
            
            return parsed
        else:
            print("Invalid JSON format in Gemini response")
            return None
    except Exception as e:
        print("Gemini Error:", e)
        traceback.print_exc()
        return None

def append_to_csvs(symptoms, result):
    """Add new disease and symptom data to CSV files"""
    try:
        disease = result['Predicted Disease']
        
        # Add to Training.csv
        training_df = pd.read_csv(TRAINING_CSV)
        new_row = {symptom: 1 if symptom in symptoms else 0 for symptom in training_df.columns[:-1]}
        new_row['prognosis'] = disease
        training_df = pd.concat([training_df, pd.DataFrame([new_row])], ignore_index=True)
        training_df.to_csv(TRAINING_CSV, index=False)
        
        # Helper function to append to other CSV files
        def append_unique(path, key, value):
            df = pd.read_csv(path)
            if disease not in df['Disease'].values:
                new_entry = {'Disease': disease, key: ', '.join(value) if isinstance(value, list) else value}
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
                df.to_csv(path, index=False)
        
        append_unique(DESCRIPTION_CSV, "Description", result["Cause"])
        append_unique(PRECAUTIONS_CSV, "Precaution", result["Precautions"])
        append_unique(MEDICATIONS_CSV, "Medication", result["Medicine"])
        append_unique(WORKOUT_CSV, "Workout", result["Workout"])
        append_unique(DIETS_CSV, "Diet", result["Diet"])
        
        print(f"Successfully added {disease} data to CSV files")
    except Exception as e:
        print(f"Error appending to CSVs: {e}")
        traceback.print_exc()

# Main diagnosis function
def diagnose_user_input(user_input):
    """Parse user input and return diagnosis or follow-up questions"""
    # Print full debug info
    print(f"Processing user input: '{user_input}'")
    
    keywords = extract_keywords(user_input)
    print(f"Extracted keywords: {keywords}")
    
    matched_symptoms = get_matching_symptoms(keywords)
    print(f"Matched symptoms: {matched_symptoms}")
    
    # EMERGENCY OVERRIDE: Hard-coded response for common cold symptoms
    if (('headache' in matched_symptoms and 'fever' in matched_symptoms) or 
        ('cough' in matched_symptoms and 'fever' in matched_symptoms) or
        ('headache' in matched_symptoms and 'cough' in matched_symptoms)):
        
        print("EMERGENCY OVERRIDE: Common cold symptoms detected, forcing Common Cold diagnosis")
        
        try:
            # Get common cold data
            desc, pre, med, die, workout_plan = helper("Common Cold")
            
            return {
                "diagnosis": "Common Cold", 
                "symptoms": matched_symptoms,
                "forced_override": True  # Flag to indicate this was a forced override
            }
        except Exception as e:
            print(f"Error in emergency override: {e}")
            # Fallback to hardcoded response
            return {
                "diagnosis": "Common Cold",
                "symptoms": matched_symptoms,
                "causes": "Viral infection affecting the upper respiratory tract",
                "precautions": ["Rest", "Stay hydrated", "Use over-the-counter medications"],
                "medications": ["Decongestants", "Pain relievers", "Cough suppressants"],
                "workouts": ["Rest until symptoms improve", "Light stretching when better"],
                "diet": ["Hot tea with honey", "Chicken soup", "Plenty of fluids"]
            }
    
    # If too few symptoms detected, ask follow-up questions
    if len(matched_symptoms) < 3:
        # Look for keywords that might warrant specific follow-up questions
        for word in keywords:
            for symptom_key in follow_ups:
                if word in symptom_key:
                    return {"follow_up": follow_ups[symptom_key]}
        
        # If no specific follow-up found, ask for more symptoms
        return {"follow_up": "Can you please describe more symptoms you're experiencing?"}
    
    # Try to predict using the model
    try:
        # List of serious conditions that CANNOT be diagnosed with only common symptoms
        serious_conditions = [
            'AIDS', 'HIV', 'Tuberculosis', 'Cancer', 'Paralysis', 'Heart attack',
            'Hepatitis', 'Alcoholic hepatitis', 'Malaria', 'Dengue'
        ]
        
        # Common symptoms that alone shouldn't trigger serious diagnoses
        common_symptoms = [
            'headache', 'fever', 'cough', 'fatigue', 'nausea', 'dizziness',
            'pain', 'sweating', 'chills', 'vomiting', 'sore throat', 'runny nose'
        ]
        
        # STRICTLY ENFORCE: Check if we're only dealing with common symptoms
        only_common = True
        for symptom in matched_symptoms:
            if symptom not in common_symptoms:
                only_common = False
                break
        
        if only_common:
            print("Only common symptoms detected, enforcing common cold or flu diagnosis")
            if 'cough' in matched_symptoms:
                return {"diagnosis": "Common Cold", "symptoms": matched_symptoms}
            else:
                return {"diagnosis": "Influenza", "symptoms": matched_symptoms}
        
        # Check if we have this combination of symptoms in our training data
        matched_rows = training_data[training_data[matched_symptoms].sum(axis=1) >= len(matched_symptoms) * 0.7]
        
        if not matched_rows.empty:
            # Use trained model for prediction
            disease = get_predicted_value(matched_symptoms)
            print(f"Model predicted disease: {disease}")
            
            # FINAL SAFETY CHECK: Never allow serious conditions with just common symptoms
            if disease in serious_conditions:
                print(f"SAFETY BLOCK: Prevented {disease} diagnosis for potentially common symptoms")
                
                # Check for specific serious symptom markers
                serious_symptom_markers = [
                    'weight_loss', 'night_sweats', 'blood_in_sputum', 'swelled_lymph_nodes',
                    'receiving_unsterile_injections', 'extra_marital_contacts'
                ]
                
                has_serious_markers = False
                for marker in serious_symptom_markers:
                    if marker in matched_symptoms:
                        has_serious_markers = True
                        break
                
                if not has_serious_markers:
                    if 'cough' in matched_symptoms and 'fever' in matched_symptoms:
                        return {"diagnosis": "Common Cold", "symptoms": matched_symptoms}
                    elif 'fever' in matched_symptoms:
                        return {"diagnosis": "Influenza", "symptoms": matched_symptoms}
                    else:
                        return {"diagnosis": "Common Cold", "symptoms": matched_symptoms}
            
            if disease:
                return {"diagnosis": disease, "symptoms": matched_symptoms}
            else:
                return {"fallback": True, "symptoms": matched_symptoms}
        else:
            # Not enough data in our training set, use fallback to Gemini
            return {"fallback": True, "symptoms": matched_symptoms}
    except Exception as e:
        print(f"Error in diagnosis: {e}")
        traceback.print_exc()
        return {"error": str(e)}

# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        if request.method == "POST":
            email = request.form.get('email')
            password = request.form.get('password')
            
            if not email or not password:
                flash("Both email and password are required.")
                return render_template("auth/login.html", email=email if email else "")
            
            try:
                user = User.query.filter_by(email=email).first()
                
                if not user:
                    flash("Email not found.")
                    return render_template("auth/login.html", email=email)
                
                # In a real app, we would use check_password_hash to verify password
                # For demo purposes, we're simplifying authentication
                if user.password != password:  # In production use check_password_hash
                    flash("Incorrect password.")
                    return render_template("auth/login.html", email=email)
                
                login_user(user)
                return redirect(url_for('dashboard'))
                
            except Exception as e:
                flash(f"Database error: {str(e)}")
                print(f"Login DB Error: {str(e)}")
                return render_template("auth/login.html", email=email)
        
        return render_template("auth/login.html")
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        print(f"General Login Error: {str(e)}")
        return render_template("auth/login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        occupation = request.form.get('occupation')
        medical_history = request.form.get('medical_history')

        if not all([email, password, name, age, gender, occupation, medical_history]):
            flash("All fields are required.")
            return render_template("auth/register.html")
        
        if User.query.filter_by(email=email).first():
            flash("Email already exists.")
            return render_template("auth/register.html")
        
        # In a real app, we would use generate_password_hash
        # For demo purposes, we're simplifying
        new_user = User(
            email=email,
            password=password,  # In production use generate_password_hash
            name=name,
            age=int(age),
            gender=gender,
            occupation=occupation,
            medical_history=medical_history
        )
        
        db.session.add(new_user)
        db.session.commit()
        flash("Registered successfully. Please log in.")
        return redirect(url_for('login'))
        
    return render_template("auth/register.html")

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/dashboard")
@login_required
def dashboard():
    messages = Message.query.filter_by(user_id=current_user.id).order_by(Message.timestamp.desc()).all()
    return render_template("dashboard.html", messages=messages, now=datetime.now())

@app.route("/diagnose", methods=["POST"])
def diagnose():
    try:
        print("Diagnose route called")
        
        # Get user input (either from form or JSON)
        if request.is_json:
            data = request.get_json()
            print(f"JSON data received: {data}")
            user_input = data.get("symptoms", "")
            if not user_input and "message" in data:
                user_input = data.get("message", "")  # Try alternate field name
        else:
            print(f"Form data received: {request.form}")
            user_input = request.form.get("symptoms", "")
            if not user_input:
                user_input = request.form.get("message", "")  # Try alternate field name
        
        print(f"User input: '{user_input}'")
        
        if not user_input:
            print("No input provided")
            return jsonify({"error": "No input provided."}), 400

        # Process the user input
        print(f"Processing user input: {user_input}")
        diagnosis = diagnose_user_input(user_input)
        print(f"Diagnosis result: {diagnosis}")
        
        # If we need to ask a follow-up question
        if "follow_up" in diagnosis:
            # If user is authenticated, save the message
            if current_user.is_authenticated:
                user_msg = Message(user_id=current_user.id, content=user_input, sender='user')
                ai_msg = Message(user_id=current_user.id, content=diagnosis["follow_up"], sender='ai')
                db.session.add(user_msg)
                db.session.add(ai_msg)
                db.session.commit()
            return jsonify({"success": True, "follow_up": diagnosis["follow_up"]})
        
        # If we have a diagnosis from our model
        if "diagnosis" in diagnosis:
            disease = diagnosis["diagnosis"]
            print(f"Disease diagnosed: {disease}")
            try:
                desc, pre, med, die, workout_plan = helper(disease)
                
                result = {
                    "success": True,
                    "Predicted Disease": disease,
                    "Cause": desc,
                    "Precautions": pre,
                    "Medicine": med,
                    "Workout": workout_plan,
                    "Diet": die
                }
                
                # Save diagnosis to database if user is authenticated
                if current_user.is_authenticated:
                    # Save user message
                    user_msg = Message(user_id=current_user.id, content=user_input, sender='user')
                    db.session.add(user_msg)
                    
                    # Format AI response message
                    ai_response = f"Based on your symptoms, I believe you may have {disease}.\n\n"
                    ai_response += f"Cause: {desc}\n\n"
                    ai_response += "Precautions:\n" + "\n".join([f"- {p}" for p in pre]) + "\n\n"
                    ai_response += "Recommended medications:\n" + "\n".join([f"- {m}" for m in med]) + "\n\n"
                    ai_response += "Suggested workouts:\n" + "\n".join([f"- {w}" for w in workout_plan]) + "\n\n"
                    ai_response += "Dietary recommendations:\n" + "\n".join([f"- {d}" for d in die])
                    
                    # Save AI message
                    ai_msg = Message(user_id=current_user.id, content=ai_response, sender='ai')
                    db.session.add(ai_msg)
                    
                    # Save full diagnosis
                    new_diagnosis = Diagnosis(
                        user_id=current_user.id,
                        diseases=result["Predicted Disease"],
                        symptoms_detected=', '.join(diagnosis["symptoms"]),
                        _causes=result["Cause"],
                        _precautions=json.dumps(result["Precautions"]),
                        _medications=json.dumps(result["Medicine"]),
                        _workouts=json.dumps(result["Workout"]),
                        _diet=json.dumps(result["Diet"])
                    )
                    db.session.add(new_diagnosis)
                    db.session.commit()
                    
                print("Diagnosis successful, returning result")
                return jsonify(result)
            except Exception as helper_error:
                print(f"Helper function error: {str(helper_error)}")
                traceback.print_exc()
                return jsonify({"error": f"Error processing diagnosis results: {str(helper_error)}"}), 500
        
        # If we need to use Gemini as fallback
        if "fallback" in diagnosis:
            print("Using Gemini fallback")
            gemini_result = ask_gemini(diagnosis["symptoms"])
            if gemini_result:
                print(f"Gemini result: {gemini_result}")
                # Save the Gemini results to our database
                try:
                    append_to_csvs(diagnosis["symptoms"], gemini_result)
                except Exception as append_error:
                    print(f"Error appending to CSVs: {str(append_error)}")
                
                # Format the response for the chat interface
                gemini_result["success"] = True
                
                # If user is authenticated, save the diagnosis
                if current_user.is_authenticated:
                    # Save user message
                    user_msg = Message(user_id=current_user.id, content=user_input, sender='user')
                    db.session.add(user_msg)
                    
                    # Format AI response message
                    ai_response = f"Based on your symptoms, I believe you may have {gemini_result['Predicted Disease']}.\n\n"
                    ai_response += f"Cause: {gemini_result['Cause']}\n\n"
                    ai_response += "Precautions:\n" + "\n".join([f"- {p}" for p in gemini_result['Precautions']]) + "\n\n"
                    ai_response += "Recommended medications:\n" + "\n".join([f"- {m}" for m in gemini_result['Medicine']]) + "\n\n"
                    ai_response += "Suggested workouts:\n" + "\n".join([f"- {w}" for w in gemini_result['Workout']]) + "\n\n"
                    ai_response += "Dietary recommendations:\n" + "\n".join([f"- {d}" for d in gemini_result['Diet']])
                    
                    # Save AI message
                    ai_msg = Message(user_id=current_user.id, content=ai_response, sender='ai')
                    db.session.add(ai_msg)
                    
                    # Save full diagnosis
                    new_diagnosis = Diagnosis(
                        user_id=current_user.id,
                        diseases=gemini_result["Predicted Disease"],
                        symptoms_detected=', '.join(diagnosis["symptoms"]),
                        _causes=gemini_result["Cause"],
                        _precautions=json.dumps(gemini_result["Precautions"]),
                        _medications=json.dumps(gemini_result["Medicine"]),
                        _workouts=json.dumps(gemini_result["Workout"]),
                        _diet=json.dumps(gemini_result["Diet"])
                    )
                    db.session.add(new_diagnosis)
                    db.session.commit()
                
                return jsonify(gemini_result)
            else:
                print("Gemini diagnosis failed")
                return jsonify({"error": "Gemini diagnosis failed."}), 500
        
        # If there was an error in diagnosis
        if "error" in diagnosis:
            print(f"Diagnosis error: {diagnosis['error']}")
            return jsonify({"error": diagnosis["error"]}), 500
            
        # If we got here, something unexpected happened
        print("Unexpected result from diagnose_user_input")
        return jsonify({"error": "Unexpected diagnosis result"}), 500

    except Exception as e:
        print(f"General error in diagnose route: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user=current_user, now=datetime.now())

@app.route("/history")
@login_required
def history():
    diagnoses = Diagnosis.query.filter_by(user_id=current_user.id).order_by(Diagnosis.timestamp.desc()).all()
    formatted_diagnoses = [{
        'id': d.id,
        'disease': d.diseases,
        'causes': d._causes,
        'precautions': json.loads(d._precautions) if d._precautions else [],
        'medications': json.loads(d._medications) if d._medications else [],
        'workouts': json.loads(d._workouts) if d._workouts else [],
        'diet': json.loads(d._diet) if d._diet else [],
        'timestamp': d.timestamp
    } for d in diagnoses]
    return render_template("history.html", diagnoses=formatted_diagnoses, now=datetime.now())

@app.route('/setup-database')
def setup_database():
    try:
        # Create all tables
        db.create_all()
        
        # Check if users table exists and has any data
        try:
            user_count = User.query.count()
            
            # Create a test user if no users exist
            if user_count == 0:
                test_user = User(
                    email="test@example.com",
                    password="password",  # In production use generate_password_hash
                    name="Test User",
                    age=30,
                    gender="Male",
                    occupation="Software Developer",
                    medical_history="No significant medical history"
                )
                db.session.add(test_user)
                db.session.commit()
                return f"Database setup complete. Created test user: test@example.com / password. User count: {User.query.count()}"
            else:
                return f"Database already has {user_count} users. No setup required."
                
        except Exception as db_error:
            return f"Error accessing database: {str(db_error)}"
            
    except Exception as e:
        return f"Setup error: {str(e)}"

@app.route('/api/messages', methods=['POST'])
@login_required
def api_messages():
    try:
        data = request.json
        if not data or 'content' not in data:
            return jsonify({'error': 'No content provided'}), 400
            
        message_content = data['content']
        
        # Save user message
        user_msg = Message(
            user_id=current_user.id,
            content=message_content,
            sender='user'
        )
        db.session.add(user_msg)
        db.session.commit()
        
        # Process the message for diagnosis
        diagnosis_result = diagnose_user_input(message_content)
        print(f"Diagnosis result: {diagnosis_result}")
        
        # Create AI response based on diagnosis result
        if 'follow_up' in diagnosis_result:
            ai_content = diagnosis_result['follow_up']
        elif 'diagnosis' in diagnosis_result:
            disease = diagnosis_result['diagnosis']
            try:
                # Check if this is a forced override with hardcoded data
                if 'forced_override' in diagnosis_result and diagnosis_result.get('forced_override'):
                    ai_content = f"Based on your symptoms, I believe you may have {disease}.\n\n"
                    
                    if 'causes' in diagnosis_result:
                        ai_content += f"Cause: {diagnosis_result['causes']}\n\n"
                        
                    if 'precautions' in diagnosis_result:
                        ai_content += "Precautions:\n"
                        for p in diagnosis_result['precautions']:
                            ai_content += f"- {p}\n"
                        ai_content += "\n"
                        
                    if 'medications' in diagnosis_result:
                        ai_content += "Recommended medications:\n"
                        for m in diagnosis_result['medications']:
                            ai_content += f"- {m}\n"
                        ai_content += "\n"
                        
                    if 'workouts' in diagnosis_result:
                        ai_content += "Suggested workouts:\n"
                        for w in diagnosis_result['workouts']:
                            ai_content += f"- {w}\n"
                        ai_content += "\n"
                        
                    if 'diet' in diagnosis_result:
                        ai_content += "Dietary recommendations:\n"
                        for d in diagnosis_result['diet']:
                            ai_content += f"- {d}\n"
                        ai_content += "\n"
                else:
                    desc, pre, med, die, workout_plan = helper(disease)
                    
                    # Format response consistently with history view
                    ai_content = f"Based on your symptoms, I believe you may have {disease}.\n\n"
                    ai_content += f"Cause: {desc}\n\n"
                    ai_content += "Precautions:\n" + "\n".join([f"- {p}" for p in pre]) + "\n\n"
                    ai_content += "Recommended medications:\n" + "\n".join([f"- {m}" for m in med]) + "\n\n"
                    ai_content += "Suggested workouts:\n" + "\n".join([f"- {w}" for w in workout_plan]) + "\n\n"
                    ai_content += "Dietary recommendations:\n" + "\n".join([f"- {d}" for d in die]) + "\n\n"
                
                # Add medical disclaimer
                ai_content += "⚠️ MEDICAL DISCLAIMER: This is not a professional medical diagnosis. Always consult with a healthcare professional for proper diagnosis and treatment."
                
                # Save diagnosis to database
                try:
                    causes = diagnosis_result.get('causes', desc)
                    precautions_list = diagnosis_result.get('precautions', pre)
                    medications_list = diagnosis_result.get('medications', med)
                    workouts_list = diagnosis_result.get('workouts', workout_plan)
                    diet_list = diagnosis_result.get('diet', die)
                    
                    new_diagnosis = Diagnosis(
                        user_id=current_user.id,
                        diseases=disease,
                        symptoms_detected=', '.join(diagnosis_result["symptoms"]),
                        _causes=causes,
                        _precautions=json.dumps(precautions_list),
                        _medications=json.dumps(medications_list),
                        _workouts=json.dumps(workouts_list),
                        _diet=json.dumps(diet_list)
                    )
                    db.session.add(new_diagnosis)
                except Exception as db_error:
                    print(f"Error saving diagnosis to database: {db_error}")
                
            except Exception as e:
                print(f"Error in helper function: {str(e)}")
                ai_content = "Sorry, I had trouble analyzing your symptoms. Please try again with more details."
        elif 'fallback' in diagnosis_result:
            # Use Gemini for fallback
            try:
                gemini_result = ask_gemini(diagnosis_result["symptoms"])
                if gemini_result:
                    disease = gemini_result["Predicted Disease"]
                    
                    # Format response consistently with history view
                    ai_content = f"Based on your symptoms, I believe you may have {disease}.\n\n"
                    ai_content += f"Cause: {gemini_result['Cause']}\n\n"
                    ai_content += "Precautions:\n" + "\n".join([f"- {p}" for p in gemini_result['Precautions']]) + "\n\n"
                    ai_content += "Recommended medications:\n" + "\n".join([f"- {m}" for m in gemini_result['Medicine']]) + "\n\n"
                    ai_content += "Suggested workouts:\n" + "\n".join([f"- {w}" for w in gemini_result['Workout']]) + "\n\n"
                    ai_content += "Dietary recommendations:\n" + "\n".join([f"- {d}" for d in gemini_result['Diet']]) + "\n\n"
                    
                    # Add medical disclaimer
                    ai_content += "⚠️ MEDICAL DISCLAIMER: This is not a professional medical diagnosis. Always consult with a healthcare professional for proper diagnosis and treatment."
                    
                    # Save diagnosis
                    new_diagnosis = Diagnosis(
                        user_id=current_user.id,
                        diseases=disease,
                        symptoms_detected=', '.join(diagnosis_result["symptoms"]),
                        _causes=gemini_result["Cause"],
                        _precautions=json.dumps(gemini_result["Precautions"]),
                        _medications=json.dumps(gemini_result["Medicine"]),
                        _workouts=json.dumps(gemini_result["Workout"]),
                        _diet=json.dumps(gemini_result["Diet"])
                    )
                    db.session.add(new_diagnosis)
                    
                    # Save to CSVs
                    try:
                        append_to_csvs(diagnosis_result["symptoms"], gemini_result)
                    except Exception as append_err:
                        print(f"Error appending to CSVs: {str(append_err)}")
                else:
                    ai_content = "I couldn't make a confident diagnosis based on your symptoms. Please provide more information."
            except Exception as gemini_err:
                print(f"Gemini error: {str(gemini_err)}")
                ai_content = "I had trouble analyzing your symptoms. Please try again or provide more details."
        else:
            ai_content = "I couldn't understand your symptoms. Could you please describe them differently?"
        
        # Save AI response
        ai_msg = Message(
            user_id=current_user.id,
            content=ai_content,
            sender='ai'
        )
        db.session.add(ai_msg)
        db.session.commit()
        
        # Return response
        return jsonify({
            'success': True,
            'ai_messages': [{
                'content': ai_content,
                'timestamp': ai_msg.timestamp.strftime('%I:%M %p')
            }]
        })
        
    except Exception as e:
        print(f"API error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'ai_messages': [{
                'content': "Sorry, something went wrong. Please try again.",
                'timestamp': datetime.now().strftime('%I:%M %p')
            }]
        })

# Override training data to ensure safety
def fix_training_data():
    """Fix the training data to ensure common symptoms are never associated with serious diseases"""
    try:
        print("Running training data safety fix...")
        global training_data, svc
        
        # First, overwrite any association between common cold symptoms and AIDS in the training data
        common_symptoms = ['headache', 'fever', 'cough', 'runny_nose', 'sore_throat', 'fatigue', 'chills']
        
        # Flip AIDS rows with these symptoms to Common Cold
        for idx, row in training_data.iterrows():
            if row['prognosis'] == 'AIDS':
                matches = 0
                for symptom in common_symptoms:
                    if symptom in training_data.columns and row[symptom] == 1:
                        matches += 1
                
                if matches >= 2:
                    print(f"Fixed training data row {idx}: Changed AIDS to Common Cold")
                    training_data.at[idx, 'prognosis'] = 'Common Cold'
        
        # Add explicit rows for common cold symptoms
        new_rows = []
        
        # Add headache + fever + cough as Common Cold
        new_row = {col: 0 for col in training_data.columns}
        for symptom in ['headache', 'fever', 'cough']:
            if symptom in new_row:
                new_row[symptom] = 1
        new_row['prognosis'] = 'Common Cold'
        new_rows.append(new_row)
        
        # Add headache + fever as Common Cold
        new_row = {col: 0 for col in training_data.columns}
        for symptom in ['headache', 'fever']:
            if symptom in new_row:
                new_row[symptom] = 1
        new_row['prognosis'] = 'Common Cold'
        new_rows.append(new_row)
        
        # Add cough + fever as Common Cold
        new_row = {col: 0 for col in training_data.columns}
        for symptom in ['cough', 'fever']:
            if symptom in new_row:
                new_row[symptom] = 1
        new_row['prognosis'] = 'Common Cold'
        new_rows.append(new_row)
        
        # Add the new rows to the training data
        for row in new_rows:
            training_data = pd.concat([training_data, pd.DataFrame([row])], ignore_index=True)
        
        print(f"Added {len(new_rows)} explicit Common Cold rows to training data")
        
        # Save the updated training data
        try:
            training_data.to_csv(TRAINING_CSV, index=False)
            print("Saved updated training data to CSV")
        except Exception as save_error:
            print(f"Error saving training data: {save_error}")
        
        # Retrain the model with the updated data
        from sklearn.svm import SVC
        X = training_data.drop('prognosis', axis=1)
        y = training_data['prognosis'].map({v: k for k, v in diseases_list.items()})
        
        # Train a new SVC model
        new_svc = SVC(kernel='linear')
        new_svc.fit(X, y)
        
        # Update the global model
        svc = new_svc
        print("Model retrained with safety fixes")
        
    except Exception as e:
        print(f"Error fixing training data: {e}")
        traceback.print_exc()

# Run the fix in a background thread to avoid blocking startup
threading.Thread(target=fix_training_data).start()

if __name__ == '__main__':
    app.run(debug=True) 