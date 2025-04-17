import pywhatkit
import pyautogui
import time
import os
import pyttsx3
import speech_recognition as sr

import cv2
import time
import getpass
import os
import pyttsx3
from fpdf import FPDF
import requests

# Initialize voice engine
engine = pyttsx3.init('sapi5')
engine.setProperty('rate', 175)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Speak instructions
speak("This sound will guide you")
speak("A box will appear capturing your face. When ready, press C to capture your image.")

# Load age & gender detection models
ageProto = r"C:\Users\Aadis\OneDrive\Codes\CTU\emergency button\deploy_age.prototxt"
ageModel = r"C:\Users\Aadis\OneDrive\Codes\CTU\emergency button\age_net.caffemodel"
genderProto = r"C:\Users\Aadis\OneDrive\Codes\CTU\emergency button\deploy_gender.prototxt"
genderModel = r"C:\Users\Aadis\OneDrive\Codes\CTU\emergency button\gender_net.caffemodel"

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)','(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Load Haar Cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), [104, 117, 123], swapRB=False)
        
        # Predict Gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = GENDER_LIST[genderPreds[0].argmax()]
        
        # Predict Age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = AGE_BUCKETS[agePreds[0].argmax()]

        label = f"{gender}, {age}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Press C to Capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Save the captured image
        getUser = getpass.getuser()
        save_path = os.path.join(r'C:\Users\Aadis\OneDrive\Codes\CTU\emergency button', 'user.png')
        cv2.imwrite(save_path, frame)
        speak("Your image is now clicked and saved")
        speak(f"Detected: {gender}, age range {age}")
        print(f"[INFO] Saved to: {save_path}")
        print(f"[INFO] Gender: {gender}, Age: {age}")
        break

cap.release()
cv2.destroyAllWindows()

# === Step 1: Get IP and Location ===
def get_ip_location():
    try:
        ip = requests.get('https://api.ipify.org').text
        response = requests.get(f'https://ipinfo.io/{ip}/json')
        data = response.json()

        loc = data.get('loc', 'Unknown').split(',')  # latitude,longitude
        city = data.get('city', 'Unknown')
        region = data.get('region', 'Unknown')
        country = data.get('country', 'Unknown')
        org = data.get('org', 'Unknown')
        postal = data.get('postal', 'Unknown')

        map_link = f"https://www.google.com/maps?q={','.join(loc)}"

        return {
            'IP': ip,
            'City': city,
            'Region': region,
            'Country': country,
            'Org': org,
            'Postal': postal,
            'Latitude': loc[0] if len(loc) > 1 else 'Unknown',
            'Longitude': loc[1] if len(loc) > 1 else 'Unknown',
            'MapLink': map_link
        }
    except Exception as e:
        return {'error': str(e)}

# === Step 2: Create PDF with Image + Details ===
location_data = get_ip_location()
pdf_path = save_path.replace('.png', '_details.pdf')

pdf = FPDF()
pdf.add_page()

pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, txt="User Emergency Report", ln=True, align='C')

# Add captured image
pdf.image(save_path, x=10, y=25, w=60)  # Resize as needed

pdf.set_font("Arial", size=12)
pdf.ln(65)  # Move cursor below image

pdf.cell(200, 10, txt=f"Gender: {gender}", ln=True)
pdf.cell(200, 10, txt=f"Age Range: {age}", ln=True)
pdf.cell(200, 10, txt=f"Image Path: {save_path}", ln=True)

pdf.ln(10)
pdf.set_font("Arial", 'B', 14)
pdf.cell(200, 10, txt="IP & Location Info:", ln=True)

pdf.set_font("Arial", size=12)
for key, value in location_data.items():
    pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

pdf.output(pdf_path)
print(f"[INFO] PDF with image saved to: {pdf_path}")

# Text-to-speech setup
engine = pyttsx3.init('sapi5')
engine.setProperty('rate', 170)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Get WhatsApp number from voice or text
def get_whatsapp_number():
    r = sr.Recognizer()
    speak("Please say or type the WhatsApp number with country code. For example, say + nine one nine eight seven six five four three two one zero")
    try:
        with sr.Microphone() as source:
            print("Listening for number...")
            audio = r.listen(source, timeout=10)
            spoken_text = r.recognize_google(audio)
            print("You said:", spoken_text)
            # Clean number (e.g., "+ nine one nine..." → "+919...")
            digits = ''.join([c for c in spoken_text if c.isdigit() or c == '+'])
            return digits
    except Exception as e:
        speak("I couldn't hear you. Please type the number.")
        return input("Enter WhatsApp number (with country code, like +91...): ")

# Main function
def send_emergency_alert():
    number = get_whatsapp_number()
    speak(f"Sending emergency alert to {number}")

    message = (
    f"🚨 EMERGENCY ALERT 🚨\n"
    f"Someone between the age range {age} and gender {gender} needs help.\n"
    f"IP: {location_data.get('IP', 'Unknown')}\n"
    f"City: {location_data.get('City', 'Unknown')}\n"
    f"Region: {location_data.get('Region', 'Unknown')}\n"
    f"Country: {location_data.get('Country', 'Unknown')}\n"
    f"Org: {location_data.get('Org', 'Unknown')}\n"
    f"Postal: {location_data.get('Postal', 'Unknown')}\n"
    f"Latitude: {location_data.get('Latitude', 'Unknown')}\n"
    f"Longitude: {location_data.get('Longitude', 'Unknown')}\n"
    f"Map Link: {location_data.get('MapLink', 'Unknown')}\n"
    f"📄 A PDF report is generated and saved on the desktop.\n"
    f"Please check immediately!"
)



    # Calculate time 1 min from now
    hour = time.localtime().tm_hour
    minute = time.localtime().tm_min + 1

    pywhatkit.sendwhatmsg(number, message, hour, minute)
    time.sleep(25)  # Let WhatsApp Web load

    # Open the folder or PDF (so user can manually attach if needed)
    pdf_path = r"C:\Users\Aadis\OneDrive\Codes\CTU\emergency button\user_details.pdf"
    if os.path.exists(pdf_path):
        os.startfile(pdf_path)
        speak("PDF has been opened. Please send it manually if not attached automatically.")
    else:
        speak("PDF not found.")

# Run the alert system
send_emergency_alert()
