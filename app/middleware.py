# app/middleware.py
from functools import wraps
from flask import session, redirect, url_for, flash

def login_required(f):
    """
    Decorator to ensure the user is logged in.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            flash("You must be logged in to access this page.", "warning")
            return redirect(url_for("main.login"))
        return f(*args, **kwargs)
    return decorated_function

def roles_required(*allowed_roles):
    """
    Decorator to restrict access to specific user roles.
    Example: @roles_required('farmer', 'provider')
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not session.get("logged_in"):
                flash("You must be logged in to access this page.", "warning")
                return redirect(url_for("main.login"))
            
            user_role = session.get("user_role")
            if user_role not in allowed_roles:
                flash("You do not have permission to access this page.", "danger")
                return redirect(url_for("main.index"))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator