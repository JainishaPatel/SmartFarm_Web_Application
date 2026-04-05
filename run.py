from waitress import serve
from app import create_app

app = create_app()

# Print a message for local testing
print("🚀 Flask app is running locally on http://127.0.0.1:8000")

# Start Waitress server
serve(app, host="0.0.0.0", port=8000)