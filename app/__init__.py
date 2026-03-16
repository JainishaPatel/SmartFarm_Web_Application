import os
from flask import Flask, render_template
from dotenv import load_dotenv
from flask_pymongo import PyMongo
from flask_dance.contrib.google import make_google_blueprint


os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'


# Load environment variables from .env
load_dotenv()


# Create the PyMongo object globally so it can be imported in routes
mongo = PyMongo()


# Google OAuth blueprint
google_bp = make_google_blueprint(
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    scope=[
        "openid",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/userinfo.email"
    ],
    redirect_to="main.google_login_callback"  # Flask route name
)


def create_app():
    app = Flask(__name__)

    # Set secret key from .env
    app.secret_key = os.getenv("SECRET_KEY")

    # MongoDB setup
    MONGO_URI = os.getenv("MONGO_URI")
    app.config["MONGO_URI"] = MONGO_URI
    mongo.init_app(app)


    # Register blueprints
    from .routes import main
    app.register_blueprint(main)
    app.register_blueprint(google_bp, url_prefix="/login")  # Google OAuth


    # Jinja2 filter for INR formatting
    def inr_format(value):
        value = int(round(value))  # remove decimal part
        s = str(value)  # convert to string without commas
        if len(s) <= 3:
            return f"₹ {s}"

        # Indian-style formatting
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        indian_format = ','.join(parts + [last3])
        return f"₹ {indian_format}"

    app.jinja_env.filters['inr'] = inr_format


    # Global 404 handler
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template("404.html"), 404


    return app