# create_admin.py
import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from werkzeug.security import generate_password_hash

# Load .env variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
users_collection = db["users"]

# Admin credentials
ADMIN_NAME = os.getenv("ADMIN_NAME")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")  # you can change this

# Hash the password
hashed_pw = generate_password_hash(ADMIN_PASSWORD)

# Check if admin already exists
existing_admin = users_collection.find_one({"email": ADMIN_EMAIL})
if existing_admin:
    print(f"Admin already exists: {ADMIN_EMAIL}")
else:
    # Insert admin user
    users_collection.insert_one({
        "name": ADMIN_NAME,
        "email": ADMIN_EMAIL,
        "password": hashed_pw,
        "role": "admin",
        "approved": True,
        "created_at": datetime.utcnow()
    })
    print(f"Admin created successfully: {ADMIN_EMAIL}")