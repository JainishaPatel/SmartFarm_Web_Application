import pickle
import numpy as np
import pandas as pd
import requests
import io
import os
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR ||| # Suppress TensorFlow INFO and warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # optional: disable oneDNN optimization messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
import json
import cloudinary
import cloudinary.uploader
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image



from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session, current_app, flash
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
load_model = tf.keras.models.load_model
image = tf.keras.preprocessing.image
from dotenv import load_dotenv
from . import mongo
from .middleware import login_required, roles_required
from flask_dance.contrib.google import google
from bson.json_util import dumps
from .disease import disease_dic



main = Blueprint('main', __name__)



base_dir = os.path.dirname(__file__) # app/ folder

# ------------------- Load Environment Variables -------------------

# Load environment variables from .env
load_dotenv()

# Get CSV file path from .env
PRICES_UNIQUE_DATASET_PATH = os.path.join(base_dir, "data/market_price_unique.csv")
SOIL_DATASET_PATH = os.path.join(base_dir, "data/Crop_recommendation.csv")

# Map your dataset classes
classes = ['Alluvial_Soil', 'Arid_Soil', 'Black_Soil', 'Laterite_Soil', 'Mountain_Soil', 'Red_Soil', 'Yellow_Soil']

# Now you can access keys safely
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# ------------------- Load Dataset -------------------

prices_df = pd.read_csv(PRICES_UNIQUE_DATASET_PATH)
soil_df = pd.read_csv(SOIL_DATASET_PATH)


MODELS_DIR = os.path.join(base_dir, "models") # app/models folder

CROP_MODEL_PATH = os.path.join(MODELS_DIR, "crop_simple_model.pkl")
CROP_FULL_MODEL_PATH = os.path.join(MODELS_DIR, "crop_full_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "crop_encoder.pkl")
SOIL_HEALTH_MODEL_PATH = os.path.join(MODELS_DIR, "soil_health_model.pkl")
SOIL_IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "soil_model.h5")
PLANT_MODEL_PATH = os.path.join(MODELS_DIR, "plant_model.pth")
PLANT_CLASS_MAP_PATH = os.path.join(MODELS_DIR, "plant_class_map.json")
SOIL_CLASS_MAP_PATH = os.path.join(MODELS_DIR, "soil_class_map_reverse.json")

# print("Current working dir:", os.getcwd())
# print("Files in app folder:", os.listdir(base_dir))

crop_simple_model = None
crop_full_model = None
crop_encoder = None
soil_health_model  = None

try:
    # 🌱 Crop Guide Model
    with open(CROP_MODEL_PATH, "rb") as f:
        crop_simple_model = pickle.load(f)

    with open(CROP_FULL_MODEL_PATH, "rb") as f:
        crop_full_model = pickle.load(f)

    with open(ENCODER_PATH, "rb") as f:
        crop_encoder = pickle.load(f)

    # 🧪 Soil Health Model
    with open(SOIL_HEALTH_MODEL_PATH, "rb") as f:
        soil_health_model = pickle.load(f)

    with open(SOIL_CLASS_MAP_PATH) as f:
        soil_class_map = json.load(f)

    # 🖼 Soil Image CNN Model
    soil_image_model = load_model(SOIL_IMAGE_MODEL_PATH)

    print("✅ All models loaded successfully.")

except Exception as e:
    print(f"❌ Failed to load models: {e}")


    
# -------------------------- Helpers ------------------------
def convert_to_ist(utc_timestamp):
    """Convert a unix timestamp (seconds) to IST formatted time string."""
    try:
        ts = int(utc_timestamp)
        return (datetime.utcfromtimestamp(ts) + timedelta(hours=5, minutes=30)).strftime("%I:%M %p")
    except Exception:
        current_app.logger.error(str(e))
        return None

def get_farming_recommendation(weather):
    try:
        if weather.get("temp") is None:
            return None

        if weather.get("description") and "rain" in weather["description"].lower():
            return "🌧️ Avoid spraying – Rain is expected."
        elif weather["wind"] > 10:
            return "💨 Avoid spraying – High wind speed may cause pesticide drift."
        elif weather["temp"] > 35:
            return "🔥 Avoid spraying – High temperature may cause evaporation loss."
        elif "clear" in weather["description"].lower() and weather["wind"] < 5:
            return "✅ Good time to spray pesticide – Clear sky and low wind."
        elif weather["humidity"] > 85:
            return "⚠️ High humidity detected – Monitor for fungal diseases."
        else:
            return "ℹ️ Weather is moderate – Monitor conditions before spraying."
    except:
        current_app.logger.error(str(e))
        return None

def normalize_name(name):
    name = name.lower()
    name = name.replace("___", " ")
    name = name.replace("_", " ")
    name = name.replace(":", " ")
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


# ----------------- Local Marketplace routes -----------------

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
MAX_IMAGE_SIZE = 3 * 1024 * 1024  # 3 MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # Load plant model
    checkpoint = torch.load(PLANT_MODEL_PATH, map_location=device)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    
except Exception as e:
    print("❌ Plant model load failed:", e)
    model = None

try:
    with open(PLANT_CLASS_MAP_PATH) as f:
        plant_classes = json.load(f)
except:
    plant_classes = []


# IMAGE TRANSFORM
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])


# ------------------- Public pages -------------------
@main.route("/")
def index():
    features = [
        {
            "title": "Plant Disease Detection",
            "icon": "fa-leaf",
            "text": "Detect plant diseases early using AI-powered image analysis to protect crops and increase yield.",
            "link": "main.predict",
            "color": "primary"
        },
        {
            "title": "Soil Analysis",
            "icon": "fa-flask",
            "text": "Analyze soil nutrients and health to improve crop productivity.",
            "link": "main.soil_analysis",
            "color": "primary"
        },
        {
            "title": "Crop Guide",
            "icon": "fa-seedling",
            "text": "Know which crop to grow each season.",
            "link": "main.crop_guide",
            "color": "primary"
        },
        {
            "title": "Weather Updates",
            "icon": "fa-cloud-sun",
            "text": "Live weather and alerts for your region.",
            "link": "main.weather_input",
            "color": "primary"
        },
        {
            "title": "Market Rates",
            "icon": "fa-sack-dollar",
            "text": "Current mandi prices for crops.",
            "link": "main.market_rates",
            "color": "primary"
        },
        {
            "title": "Online Market",
            "icon": "fa-cart-shopping",
            "text": "Buy seeds, tools, fertilizers, and more online.",
            "link": "main.e_commerce",
            "color": "primary"
        },
        {
            "title": "Farmer Marketplace",
            "icon": "fa-store",
            "text": "Sell your crops directly to buyers.",
            "link": "main.farmer_marketplace",
            "color": "primary"
        },
        {
            "title": "Govt. Schemes",
            "icon": "fa-building-columns",
            "text": "Know about subsidies & Yojanas.",
            "link": "main.schemes",
            "color": "primary"
        },
        {
            "title": "Crop Shelter",
            "icon": "fa-tree",
            "text": "Protect your yield in extreme weather.",
            "link": "main.crop_shelter",
            "color": "primary"
        },
    ]
    return render_template("index.html", features=features)

#------------------------------------------------------------------------------------------------------------

@main.route("/about")
def about():
    return render_template("about.html")

#------------------------------------------------------------------------------------------------------------

@main.route("/contact", methods=["GET", "POST"])
@login_required
def contact():
    """
    Contact page:
    - GET: Pre-fill form with user's session info
    - POST: Validate, save to MongoDB, and show success/error messages
    """

    # Pre-fill form fields from session
    name = session.get("user_name", "")
    email = session.get("user_email", "")
    message_content = ""

    success_msg = None
    error_msg = None

    if request.method == "POST":
        # Get form data
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        message_content = request.form.get("message", "").strip()

        # Basic validation
        if not all([name, email, message_content]):
            error_msg = "❌ All fields are required."
        else:
            try:
                # Insert into MongoDB
                mongo.db.contacts.insert_one({
                    "name": name,
                    "email": email,
                    "message": message_content,
                    "created_at": datetime.utcnow()
                })
                success_msg = "✅ Your message has been sent!"
                message_content = ""  # Clear textarea after success
            except Exception as e:
                print("MongoDB insert error:", e)
                error_msg = "❌ Something went wrong while sending your message."

    return render_template(
        "contact.html",
        name=name,
        email=email,
        message=message_content,
        success=success_msg,
        error=error_msg
    )

#------------------------------------------------------------------------------------------------------------

@main.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

         # --- Basic validation ---
        if not email or not password:
            flash("⚠️ Both email and password are required.", "danger")
            return redirect(url_for("main.login"))

        # --- Fetch user from MongoDB ---
        user = mongo.db.users.find_one({"email": email})
        if not user:
            flash("⚠️ No account found with this email.", "danger")
            return redirect(url_for("main.login"))

        # --- Check password ---
        if not check_password_hash(user["password"], password):
            flash("⚠️ Incorrect password.", "danger")
            return redirect(url_for("main.login"))

        # --- Optional approval check ---
        # if user.get("approved") is False:
        #     flash("⚠️ Your account is not yet approved by admin.", "warning")
        #     return redirect(url_for("main.login"))
        
        # --- Create session ---
        session["user_email"] = user["email"]
        session["user_name"] = user["name"]
        session["user_role"] = user["role"]
        session["logged_in"] = True

        flash(f"✅ Welcome back, {user['name']}!", "success")
        return redirect(url_for("main.index"))

    # GET request
    return render_template("login.html")

#------------------------------------------------------------------------------------------------------------

@main.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        # --- Get form data ---
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "").strip().lower()

        # --- Basic validation ---
        if not all([name, email, password, role]):
            flash("⚠️ All fields are required.", "danger")
            return redirect(url_for("main.signup"))

        # --- Role validation ---
        if role not in ["farmer", "buyer", "provider"]:
            flash("⚠️ Invalid role selected.", "danger")
            return redirect(url_for("main.signup"))

        # --- Email format validation ---
        email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_regex, email):
            flash("⚠️ Invalid email format.", "danger")
            return redirect(url_for("main.signup"))

        # --- Password strength check ---
        if len(password) < 6:
            flash("⚠️ Password must be at least 6 characters.", "danger")
            return redirect(url_for("main.signup"))
        # Optional: you can add more checks (uppercase, digit, special char) here

        # --- Check if user already exists ---
        existing_user = mongo.db.users.find_one({"email": email})
        if existing_user:
            flash("⚠️ Email already registered.", "danger")
            return redirect(url_for("main.signup"))

        # --- Hash the password ---
        hashed_pw = generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)
        

        # --- Default approval: only farmers auto-approved ---
        approved = True

        # --- Insert user into MongoDB ---
        mongo.db.users.insert_one({
            "name": name,
            "email": email,
            "password": hashed_pw,
            "role": role,
            "approved": approved,
            "created_at": datetime.utcnow()
        })

        # --- Automatically log in the user ---
        session["user_email"] = email
        session["user_role"] = role
        session["user_name"] = name
        session["logged_in"] = True

        flash(f"✅ Signup successful! Welcome, {name}", "success")
        return redirect(url_for("main.index"))

    return render_template("signup.html")

#------------------------------------------------------------------------------------------------------------

@main.route("/logout")
def logout():
    # Clear session or any logout logic here
    session.clear()
    return redirect(url_for("main.index"))

from flask_dance.contrib.google import google

#------------------------------------------------------------------------------------------------------------

# ------------ Google ----------------
@main.route("/auth/google/callback")
def google_login_callback():
    # Check if the user is authorized
    if not google.authorized:
        return redirect(url_for("google.login"))  # Redirects to Google login

    # Get user info
    resp = google.get("/oauth2/v2/userinfo")
    if resp.ok:
        user_info = resp.json()
        email = user_info.get("email")
        name = user_info.get("name")

        user = mongo.db.users.find_one({"email": email})
        if not user:
            # Temporarily store in session until role is chosen
            session["temp_name"] = name
            session["temp_email"] = email
            return redirect(url_for("main.choose_role")) 

        # If user already exists, proceed normally
        session["user_name"] = user["name"]
        session["user_email"] = user["email"]
        session["user_role"] = user["role"]
        session["logged_in"] = True 


        flash(f"✅ Welcome back, {name}! Logged in as {user['role']}.", "success")
        return redirect(url_for("main.index"))

    flash("⚠️ Google login failed.", "danger")
    return redirect(url_for("main.login"))

#------------------------------------------------------------------------------------------------------------

@main.route("/choose-role", methods=["GET", "POST"])
def choose_role():
    if request.method == "POST":
        role = request.form.get("role")
        name = session.get("temp_name")
        email = session.get("temp_email")

        if role not in ["farmer", "buyer", "provider"]:
            flash("⚠️ Invalid role selected.", "danger")
            return redirect(url_for("main.choose_role"))

        # Insert user into DB with chosen role
        mongo.db.users.insert_one({
            "name": name,
            "email": email,
            "role": role,
            "approved": True,
            "created_at": datetime.utcnow()
        })

        # Save session
        session["user_name"] = name
        session["user_email"] = email
        session["user_role"] = role
        session["logged_in"] = True

        # Remove temp
        session.pop("temp_name", None)
        session.pop("temp_email", None)

        flash(f"🎉 Account created successfully! Signed in as {name} ({role}).", "success")
        return redirect(url_for("main.index"))

    return render_template("choose_role.html")

#------------------------------------------------------------------------------------------------------------



# ------------------- Protected pages (login required) -------------------

# ================== ROUTES ================== #

@main.route("/crop_guide", methods=["GET", "POST"])
@login_required
def crop_guide():
    if request.method == "POST":
        if crop_simple_model is None or crop_encoder is None:
            # inside route, current_app is safe
            current_app.logger.error("Model or encoder not loaded for /predict")
            return render_template(
                "crop_guide.html",
                prediction=None,
                error="Model not available on server."
            )

        try:
            temp_raw = request.form.get("temperature")
            hum_raw = request.form.get("humidity")

            if temp_raw is None or hum_raw is None:
                return render_template(
                    "crop_guide.html",
                    prediction=None,
                    error="Temperature & humidity required."
                )

            temperature = float(temp_raw)
            humidity = float(hum_raw)

            current_app.logger.info(f"Input Data - Temp: {temperature}, Humidity: {humidity}")

            input_data = pd.DataFrame([[temperature, humidity]], columns=["temperature", "humidity"])
            crop_index = crop_simple_model.predict(input_data)[0]
            crop_name = crop_encoder.inverse_transform([crop_index])[0]

            return render_template("crop_guide.html", prediction=crop_name)

        except Exception as e:
            current_app.logger.error(f"Error in prediction: {e}")
            return render_template("crop_guide.html", prediction=None, error=str(e))

    return render_template("crop_guide.html", prediction=None)

#--------------------------------------------------------------------------------------------------------------

@main.route("/get_api_key", methods=["GET"])
@login_required
def get_api_key():
    return jsonify({"error": "Not allowed in production"}), 403

#------------------------------------------------------------------------------------------------------------

@main.route("/predict_auto", methods=["POST"])
@login_required
def predict_auto():
    if crop_simple_model is None or crop_encoder is None:
        current_app.logger.error("Model or encoder not loaded for /predict_auto")
        return jsonify({"error": "Model not available on server."}), 500

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "JSON body expected"}), 400

        # Try to get latitude & longitude from client
        lat = data.get("lat")
        lon = data.get("lon")
        fertilizer = data.get("fertilizer", 0)  # optional

        if lat is None or lon is None:
            return jsonify({"error": "lat and lon required"}), 400

        # Fetch weather from OpenWeatherMap using server-side API key
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(weather_url, timeout=10)
        res.raise_for_status()
        w = res.json()

        temperature = w["main"]["temp"]
        humidity = w["main"]["humidity"]

        current_app.logger.info(f"Auto Prediction - Temp: {temperature}, Humidity: {humidity}, Fertilizer: {fertilizer}")

        # Prepare input for model (adjust column names to your model)
        input_data = pd.DataFrame([[temperature, humidity]], columns=["temperature", "humidity"])
        crop_index = crop_simple_model.predict(input_data)[0]
        crop_name = crop_encoder.inverse_transform([crop_index])[0]

        return jsonify({
            "prediction": crop_name,
            "temperature": temperature,
            "humidity": humidity
        })

    except Exception as e:
        current_app.logger.error(f"Error in auto prediction: {e}")
        return jsonify({"error": str(e)}), 500
    
#------------------------------------------------------------------------------------------------------------

@main.route("/weather_input")
@login_required
def weather_input():
    try:
        ip_info = requests.get("https://ipapi.co/json/").json()
        city = ip_info.get("city", "Delhi")
        return redirect(url_for("main.weather", city=city))
    except Exception:
        return redirect(url_for("main.weather", city="Delhi"))

#------------------------------------------------------------------------------------------------------------

@main.route("/weather_redirect", methods=["GET"])
@login_required
def weather_redirect():
    city = request.args.get("city")
    if city:
        return redirect(url_for("main.weather", city=city))
    return redirect(url_for("main.weather", city="Delhi"))  # fallback

#------------------------------------------------------------------------------------------------------------

@main.route('/weather/<city>')
@login_required
def weather(city):
    api_key = os.getenv("WEATHER_API_KEY")  # Replace with your OpenWeatherMap API key
    weather_data = {}
    

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()

        if data.get("cod") != 200:
            raise Exception("API error")

        weather_data = {
            "city": city,
            "temp": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"].title(),
            "wind": data["wind"]["speed"],
            "icon": data["weather"][0]["icon"], # <-- THIS is what powers the icon image!
            "country": data["sys"]["country"],  # Add country
            # Convert sunrise and sunset times to IST
            "sunrise_time": convert_to_ist(data["sys"]["sunrise"]),
            "sunset_time": convert_to_ist(data["sys"]["sunset"]),
            "date": datetime.fromtimestamp(data["dt"]).strftime('%d-%m-%Y'),
            "time": datetime.fromtimestamp(data["dt"]).strftime('%I:%M %p'),
        }

        # Generate smart farming advice
        recommendation = get_farming_recommendation(weather_data)

    except Exception:
        weather_data = {"error": "Unable to fetch weather data"}

    return render_template("weather.html", weather=weather_data, recommendation=recommendation)

#------------------------------------------------------------------------------------------------------------

@main.route("/api/weather")
@login_required
def get_weather():
    try:
        API_KEY = os.getenv("WEATHER_API_KEY")  # Replace with your actual key

        if not API_KEY:
            return jsonify({"error": "API key not found."}), 500
        
        url = f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q=auto:ip"
        response = requests.get(url)

        if response.status_code != 200:
            return jsonify({"error": f"Weather API returned status {response.status_code}"}), response.status_code

        return jsonify(response.json())
    except Exception as e:
        print("Weather API error:", e)
        return jsonify({"error": str(e)}), 500

#--------------------------------------------------------------------------------------

@main.route("/schemes")
@login_required
def schemes():  
    # Logic to display government schemes and subsidies
    return render_template("schemes.html")

#--------------------------------------------------------------------------------------

@main.route("/crop_shelter")
def crop_shelter():
    return render_template("crop_shelter.html")

#--------------------------------------------------------------------------------------

@main.route("/orders")
@login_required
def orders():
    """
    Display all orders of the currently logged-in user.
    """
    user_email = session.get("user_email")
    
    if not user_email:
        flash("⚠️ Please log in to view your orders.", "danger")
        return redirect(url_for("main.login"))

    # Fetch user's orders from MongoDB
    orders = list(mongo.db.orders.find({"user_email": user_email}).sort("created_at", -1))
    
    # Convert ObjectId and datetime to string for template
    for order in orders:
        order["_id"] = str(order["_id"])
        order["created_at"] = order["created_at"].strftime("%d-%m-%Y %I:%M %p")

    return render_template("orders.html", orders=orders)

#-------------------------------------------------------------------------------------

@main.route("/profile")
def profile():
    if not session.get("logged_in"):
        return redirect(url_for("main.login"))

    user = mongo.db.users.find_one(
        {"email": session.get("user_email")}
    )

    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("main.login"))

    return render_template("profile.html", user=user)

#--------------------------------------------------------------------------------------

@main.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html", response=None)


@main.route('/prompt', methods=['POST'])
@login_required
def prompt():
    prompt = request.form.get("prompt", "")

    template = {
        "model": "gemma:2b",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post('http://127.0.0.1:11434/api/generate', json=template)
    llm_response = response.json()


    # Ollama puts the text in 'response' key at top level
    bot_answer = llm_response.get("response", "No response from model")


    return render_template("chatbot.html", response=bot_answer)

#------------------------------------------------------------------------------------------------------------

@main.route("/market_rates", methods=["GET", "POST"])
def market_rates():
    state = district = commodity = None

    # Dropdown lists (safe, small)
    states = sorted(prices_df["STATE"].dropna().unique())
    districts = sorted(prices_df["District Name"].dropna().unique())
    commodities = sorted(prices_df["Commodity"].dropna().unique())

    # 🔥 IMPORTANT: empty by default
    rates = []

    if request.method == "POST":
        rates_df = prices_df.copy()

        state = request.form.get("state")
        district = request.form.get("district")
        commodity = request.form.get("commodity")

        # Apply filters
        if state:
            rates_df = rates_df[rates_df["STATE"] == state]
        if district:
            rates_df = rates_df[rates_df["District Name"] == district]
        if commodity:
            rates_df = rates_df[rates_df["Commodity"] == commodity]

        # OPTIONAL: limit rows (extra safety)
        rates_df = rates_df.head(30)

        rates = rates_df.to_dict(orient="records")

    return render_template(
        "market_rates.html",
        states=states,
        districts=districts,
        commodities=commodities,
        state=state,
        district=district,
        commodity=commodity,
        rates=rates
    )

#------------------------------------------------------------------------------------------------------------

# ----------------- E-Commerce -----------------

# Main e-commerce page
@main.route('/e_commerce')
@login_required
def e_commerce():
    products = [
        {"name":"Seeds","icon":"fa-seedling","color":"text-success","types":["Wheat","Rice","Maize"]},
        {"name":"Fertilizers","icon":"fa-bottle-water","color":"text-warning","types":["Urea","DAP","NPK"]},
        {"name":"Pesticides","icon":"fa-skull-crossbones","color":"text-dark","types":["Insecticide","Fungicide"]}
    ]
    return render_template('e_commerce.html', products=products)

#------------------------------------------------------------------------------------------------------------

# Product type page (shows images)
@main.route('/product/<product_name>', methods=['GET'])
@login_required
def product_types(product_name):
    
    # Seed categories
    seed_categories = {
        "Cereals": ["Wheat", "Rice", "Maize", "Barley", "Millet"],
        "Pulses": ["Lentil", "Chickpea", "Green Gram", "Black Gram"],
        "Oilseeds": ["Groundnut", "Mustard", "Soybean", "Sesame"],
        "Vegetables": ["Tomato", "Onion", "Potato", "Brinjal"],
        "Fruits": ["Mango", "Banana", "Papaya", "Guava"]
    }

    if product_name.lower() == "seeds":
        types = seed_categories
    else:
        product_data = {
            "Fertilizers": ["Urea", "DAP", "NPK", "Compost"],
            "Pesticides": ["Herbicide", "Insecticide", "Fungicide"]
        }
        types = {"Other": product_data.get(product_name, [])}

    return render_template(
        "product_types.html",
        product_name=product_name,
        types=types
    )

#------------------------------------------------------------------------------------------------------------

# Order form submission
@main.route('/order', methods=['POST'])
@login_required
def order_form():
    # Get form data
    product = request.form.get('product')
    subtype = request.form.get('subtype')
    name = request.form.get('name')
    phone = request.form.get('phone')
    email = request.form.get('email')  # NEW: fetch from form
    address = request.form.get('address')
    quantity = request.form.get('quantity')
    unit = request.form.get('unit')  # make sure unit is sent from form

    # Basic validation
    if not all([product, subtype, name, phone, email, address, quantity, unit]):
        flash("⚠️ Please fill all the fields to place an order.", "danger")
        return redirect(request.referrer or url_for('main.e_commerce'))

    # Convert quantity safely
    try:
        quantity_value = float(quantity)
        if quantity_value <= 0:
            flash("⚠️ Quantity must be greater than 0.", "danger")
            return redirect(request.referrer or url_for('main.e_commerce'))
    except ValueError:
        flash("⚠️ Quantity must be a number.", "danger")
        return redirect(request.referrer or url_for('main.e_commerce'))
    
    # Price per unit mapping
    unit_prices = {
        "Seeds": {"kg": 50, "packet": 10, "bag": 400},
        "Fertilizers": {"kg": 60, "packet": 15, "bag": 450},
        "Pesticides": {"ml": 5, "L": 400, "bottle": 250}
    }

    # Get price per unit and total price
    price_per_unit = unit_prices.get(product, {}).get(unit, 0)
    total_price = price_per_unit * quantity_value

    # Prepare order data
    order_data = {
        "user_email": email,  # now taken from form
        "product": product,
        "subtype": subtype,
        "name": name,
        "phone": phone,
        "address": address,
        "quantity": quantity_value,
        "unit": unit,
        "price_per_unit": price_per_unit,
        "total_price": total_price,
        "status": "Booked",
        "created_at": datetime.utcnow()
    }

    # Save order to MongoDB with error handling
    try:
        mongo.db.orders.insert_one(order_data)
        flash(f"✅ Your order for {quantity} {unit} of {subtype} ({product}) has been booked successfully!", "success")
    except Exception as e:
        flash("❌ Something went wrong while placing your order. Please try again.", "danger")
        print("Order insert error:", e)

    # Redirect back to the product page
    return redirect(request.referrer or url_for('main.e_commerce'))
#------------------------------------------------------------------------------------------------------------

# ------------------- Marketplace Page -------------------
@main.route("/farmer_marketplace")
@login_required
def farmer_marketplace():
    """
    Renders the Local Marketplace page.
    Listings will be loaded asynchronously via JS from /api/listings
    """
    return render_template("farmer_marketplace.html")

#------------------------------------------------------------------------------------------------------------

# ------------------- API for Listings -------------------
@main.route("/api/listings", methods=["GET"])
@login_required
def api_listings():
    """
    Returns JSON list of marketplace listings.
    Optional query parameters:
    - q: search query (crop_name, seller_name, location)
    - location: filter by location
    - seller_email: filter by seller
    - limit: maximum results (default 100)
    """
    try:
        q = request.args.get("q", "").strip().lower()
        location = request.args.get("location", "").strip().lower()
        seller_email = request.args.get("seller_email")
        limit = int(request.args.get("limit", 100))

        query = {}

        # Text search across crop_name, seller_name, location
        if q:
            query["$or"] = [
                {"crop_name": {"$regex": q, "$options": "i"}},
                {"seller_name": {"$regex": q, "$options": "i"}},
                {"location": {"$regex": q, "$options": "i"}}
            ]
        if location:
            query["location"] = {"$regex": location, "$options": "i"}
        if seller_email:
            query["seller_email"] = seller_email

        # Fetch listings from MongoDB
        docs = list(mongo.db.marketplace.find(query).sort("created_at", -1).limit(limit))

        # Convert ObjectId & datetime to JSON-friendly
        for d in docs:
            d["_id"] = str(d["_id"])
            if d.get("created_at"):
                d["created_at"] = d["created_at"].isoformat()

        return jsonify(json.loads(dumps(docs)))

    except Exception as e:
        print("API listings error:", e)
        return jsonify({"error": "Server error"}), 500

#------------------------------------------------------------------------------------------------------------

# ------------------- Add Listing Page -------------------
@main.route("/add_listing", methods=["GET", "POST"])
@login_required
@roles_required("farmer")
def add_listing():
    """
    Farmers can post a listing.
    Handles image upload and inserts document into MongoDB.
    """
    # user_role = session.get("user_role", "")
    # if user_role != "farmer":
    #     flash("Only farmers can post listings.", "danger")
    #     return redirect(url_for("main.farmer_marketplace"))

    if request.method == "POST":
        crop_name = request.form.get("crop_name", "").strip()
        quantity = request.form.get("quantity", "").strip()
        unit = request.form.get("unit", "kg")  # default kg
        price = request.form.get("price", "").strip()
        contact = request.form.get("contact", "").strip()
        location = request.form.get("location", "").strip()
        description = request.form.get("description", "").strip()

        # Inside add_listing
        public_id = None
        file = request.files.get("image")
        if file and file.filename != "":
            upload_result = cloudinary.uploader.upload(file, folder="farmer_marketplace")
            image_url = upload_result.get("secure_url")
            public_id = upload_result.get("public_id")
        else:
            image_url = url_for("static", filename="assets/default_crop.jpg")


        doc = {
            "crop_name": crop_name,
            "quantity": quantity,
            "unit": unit,
            "price": price,
            "contact": contact,
            "location": location,
            "description": description,
            "image_url": image_url,
            "public_id": public_id,  
            "seller_name": session.get("user_name", "Anonymous"),
            "seller_email": session.get("user_email"),
            "created_at": datetime.utcnow()
        }

        mongo.db.marketplace.insert_one(doc)
        flash("Listing added successfully.", "success")
        return redirect(url_for("main.farmer_marketplace"))

    return render_template(
        "add_listing.html",
        seller_name=session.get("user_name", ""),
        seller_contact=session.get("user_email", "")  # prefill contact if stored
    )

#------------------------------------------------------------------------------------------------------------

# ------------------- My Listings -------------------
@main.route("/my_listings")
@login_required
@roles_required("farmer")
def my_listings():
    """Shows listings posted by current user"""
    user_email = session.get("user_email")
    docs = list(mongo.db.marketplace.find({"seller_email": user_email}).sort("created_at", -1))
    for d in docs:
        d["_id"] = str(d["_id"])
        if d.get("created_at"):
            d["created_at"] = d["created_at"].isoformat()
    return render_template("my_listings.html", listings=docs)

#------------------------------------------------------------------------------------------------------------

# ------------------- Delete Listing -------------------
@main.route("/delete_listing/<listing_id>", methods=["POST"])
@login_required
@roles_required("farmer")
def delete_listing(listing_id):
    """Delete a listing (only by seller or admin)"""
    try:
        doc = mongo.db.marketplace.find_one({"_id": ObjectId(listing_id)})
        if not doc:
            flash("Listing not found.", "danger")
            return redirect(url_for("main.my_listings"))

        user_email = session.get("user_email")
        user_role = session.get("user_role", "")

        if doc.get("seller_email") != user_email and user_role != "admin":
            flash("Not authorized to delete this listing.", "danger")
            return redirect(url_for("main.my_listings"))

        # Remove image from Cloudinary
        public_id = doc.get("public_id")
        if public_id:
            try:
                cloudinary.uploader.destroy(public_id)
            except Exception as e:
                print("Cloudinary delete failed:", e)

        # Delete document from MongoDB
        mongo.db.marketplace.delete_one({"_id": ObjectId(listing_id)})

        flash("Listing deleted successfully.", "success")
        return redirect(url_for("main.my_listings"))

    except Exception as e:
        print("Error deleting listing:", e)
        flash("Error deleting listing.", "danger")
        return redirect(url_for("main.my_listings"))
    
#------------------------------------------------------------------------------------------------------------

@main.route('/soil_analysis', methods=['GET', 'POST'])
@login_required
def soil_analysis():
    result = None
    image_url = None

    # ADD THIS BLOCK (same as soil_health route)
    min_max = {
        "N": {"min": int(soil_df["N"].min()), "max": int(soil_df["N"].max())},
        "P": {"min": int(soil_df["P"].min()), "max": int(soil_df["P"].max())},
        "K": {"min": int(soil_df["K"].min()), "max": int(soil_df["K"].max())},
        "temperature": {"min": float(soil_df["temperature"].min()), "max": float(soil_df["temperature"].max())},
        "humidity": {"min": float(soil_df["humidity"].min()), "max": float(soil_df["humidity"].max())},
        "ph": {"min": float(soil_df["ph"].min()), "max": float(soil_df["ph"].max())}
    }

    if request.method == 'POST':
        file = request.files.get('soil_image')

        if file and file.filename != "":

            if not allowed_file(file.filename):
                return "Invalid file type"

            if soil_image_model is None:
                return "Soil model not loaded"
            
            try: 
                img_bytes = file.read()
                img = image.load_img(io.BytesIO(img_bytes), target_size=(128,128))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = soil_image_model.predict(img_array, verbose=0)
                class_idx = np.argmax(prediction, axis=1)[0]
                
                result = soil_class_map[str(class_idx)]

                # Upload image
                file.seek(0)
                upload_result = cloudinary.uploader.upload(file)
                image_url = upload_result['secure_url']
            
            except Exception as e:
                print("❌ Error:", e)
                result = "Error processing image"

    return render_template(
        'soil_analysis.html',   # make sure this matches template name
        result=result,
        image_url=image_url,
        min_max=min_max       # VERY IMPORTANT
    )

#-----------------------------------------------

@main.route('/predict_health', methods=["POST"])
@login_required
def predict_health():
    data = request.json
    try:
        features = [[
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph'])
        ]]

        # Predict Soil Health
        health_prediction = soil_health_model.predict(features)[0]

        # Predict Crop
        crop_prediction = crop_full_model.predict(features)[0]

        return jsonify({
            "soil_health": health_prediction,
            "recommended_crop": crop_prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#------------------------------------------------------------------------------------------------------------

@main.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'GET':
        return render_template('disease.html')

    # Check file exists
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files.get('file')

    if file.filename == '':
        return "No selected file"

    # Validate file type
    if not allowed_file(file.filename):
        return "Invalid file type (only images allowed)"

    # Check model loaded
    if model is None or not plant_classes:
        return "Model not available on server"

    try:
        # Secure filename (optional but good practice)
        filename = secure_filename(file.filename)

        # Upload to Cloudinary (safe handling)
        try:
            upload_result = cloudinary.uploader.upload(file, folder="plant_disease")
            image_url = upload_result.get("secure_url")
        except Exception as e:
            current_app.logger.error(f"Cloudinary error: {e}")
            image_url = None  # fallback

        # Reset file pointer
        file.stream.seek(0)

        # Image preprocessing
        img = Image.open(file.stream).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        # Prediction
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)

        _, pred = torch.max(outputs, 1)
        pred_idx = pred.item()

        # Safe index check
        if pred_idx >= len(plant_classes):
            return "Prediction error: class index out of range"

        result = plant_classes[pred_idx]
        confidence = probs[0][pred_idx].item()

        # Safe split
        if "___" in result:
            crop, disease = result.split("___")
        else:
            crop, disease = "Unknown", result

        disease = disease.replace("_", " ")

        # Disease info
        disease_info = disease_dic.get(result, "No information available.")

        return render_template(
            'disease.html',
            crop=crop,
            disease=disease,
            confidence=round(confidence * 100, 2),
            disease_info=disease_info,
            image_url=image_url
        )

    except Exception as e:
        current_app.logger.error(f"Prediction error: {e}")
        return render_template(
            'disease.html',
            error="Something went wrong during prediction."
        )