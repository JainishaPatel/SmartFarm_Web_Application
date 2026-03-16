import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from werkzeug.security import generate_password_hash

# Load environment variables
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
users_collection = db["users"]

# ================= USERS LIST =================
users = [
    {
        "name": "Jack",
        "email": "Jack72@gmail.com",
        "password": "Jack72",
        "role": "buyer"
    },
    {
        "name": "Jenny",
        "email": "Jenny15@gmail.com",
        "password": "Janny15",
        "role": "provider"
    },
    {
        "name": "John",
        "email": "John04@gmail.com",
        "password": "John04",
        "role": "farmer"
    }
]
# ==============================================

for u in users:
    # Check if user already exists
    if users_collection.find_one({"email": u["email"]}):
        print(f"User already exists: {u['email']}")
        continue

    # Insert user
    users_collection.insert_one({
        "name": u["name"],
        "email": u["email"],
        "password": generate_password_hash(u["password"]),
        "role": u["role"],
        "approved": True,
        "created_at": datetime.utcnow()
    })

    print(f"✅ User created: {u['email']} ({u['role']})")

print("Bulk user creation completed.")
