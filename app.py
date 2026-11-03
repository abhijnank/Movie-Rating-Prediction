from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os
import pandas as pd
import numpy as np

from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "movie_rating_project_secret_key"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE = os.path.join(BASE_DIR, "movie_rating.db")
DATASET = os.path.join(BASE_DIR, "rating.csv")


# --------------------------------------------------
# MACHINE LEARNING CONFIGURATION
# --------------------------------------------------

FEATURES = [
    "Genre",
    "Runtime_Min",
    "Budget_Million",
    "Number_of_Votes",
    "Previous_Rating"
]

NUMERIC_FEATURES = [
    "Runtime_Min",
    "Budget_Million",
    "Number_of_Votes",
    "Previous_Rating"
]

CATEGORICAL_FEATURES = ["Genre"]

TARGET = "Rating"

model = None

mae = 0
mse = 0
rmse = 0
r2 = 0
training_rows = 0


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


# --------------------------------------------------
# CREATE DATABASE TABLES
# --------------------------------------------------

def create_tables():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            genre TEXT,
            runtime REAL,
            budget REAL,
            votes INTEGER,
            previous_rating REAL,
            predicted_rating REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Add missing columns if an older database is used.
    cursor.execute("PRAGMA table_info(users)")
    user_columns = [row["name"] for row in cursor.fetchall()]

    if "role" not in user_columns:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'"
        )

    cursor.execute("PRAGMA table_info(predictions)")
    prediction_columns = [row["name"] for row in cursor.fetchall()]

    required_columns = {
        "genre": "TEXT",
        "runtime": "REAL",
        "budget": "REAL",
        "votes": "INTEGER",
        "previous_rating": "REAL",
        "predicted_rating": "REAL",
        "created_at": "TIMESTAMP"
    }

    for column, data_type in required_columns.items():
        if column not in prediction_columns:
            cursor.execute(
                f"ALTER TABLE predictions ADD COLUMN {column} {data_type}"
            )

    # Default admin account
    cursor.execute(
        "SELECT id FROM users WHERE username = ?",
        ("admin",)
    )

    if cursor.fetchone() is None:

        cursor.execute("""
            INSERT INTO users (username, email, password, role)
            VALUES (?, ?, ?, ?)
        """, (
            "admin",
            "admin@movierating.com",
            generate_password_hash("Admin@123"),
            "admin"
        ))

    conn.commit()
    conn.close()


# --------------------------------------------------
# TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

def train_model():

    global model, mae, mse, rmse, r2, training_rows

    try:

        if not os.path.exists(DATASET):
            print("ERROR: rating.csv file not found.")
            return

        df = pd.read_csv(DATASET)

        required_columns = FEATURES + [TARGET]

        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            print("Missing dataset columns:", missing_columns)
            return

        df = df[required_columns].copy()

        for column in NUMERIC_FEATURES + [TARGET]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df["Genre"] = df["Genre"].astype(str).str.strip()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        df = df[df["Genre"] != ""]

        if len(df) < 5:
            print("ERROR: Dataset needs at least 5 valid rows.")
            return

        X = df[FEATURES]
        y = df[TARGET]

        preprocessing = ColumnTransformer(
            transformers=[
                (
                    "genre",
                    OneHotEncoder(handle_unknown="ignore"),
                    CATEGORICAL_FEATURES
                ),
                (
                    "numeric",
                    "passthrough",
                    NUMERIC_FEATURES
                )
            ]
        )

        pipeline = Pipeline(steps=[
            ("preprocessing", preprocessing),
            ("regression", LinearRegression())
        ])

        # Evaluate model on a test split
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        evaluation_model = Pipeline(steps=[
            ("preprocessing", ColumnTransformer(
                transformers=[
                    (
                        "genre",
                        OneHotEncoder(handle_unknown="ignore"),
                        CATEGORICAL_FEATURES
                    ),
                    (
                        "numeric",
                        "passthrough",
                        NUMERIC_FEATURES
                    )
                ]
            )),
            ("regression", LinearRegression())
        ])

        evaluation_model.fit(X_train, y_train)

        predictions = evaluation_model.predict(X_test)

        mae = float(mean_absolute_error(y_test, predictions))
        mse = float(mean_squared_error(y_test, predictions))
        rmse = float(np.sqrt(mse))

        if len(y_test) > 1:
            r2 = float(r2_score(y_test, predictions))
        else:
            r2 = 0.0

        # Train final model using complete dataset
        pipeline.fit(X, y)

        model = pipeline
        training_rows = len(df)

        print("Model trained successfully!")
        print("Dataset rows:", training_rows)
        print("MAE:", mae)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R2 Score:", r2)

    except Exception as error:
        print("Model training error:", error)


# --------------------------------------------------
# LOGIN REQUIRED DECORATOR
# --------------------------------------------------

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            flash("Please login first.")
            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return wrapper


# --------------------------------------------------
# ADMIN REQUIRED DECORATOR
# --------------------------------------------------

def admin_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            return redirect(url_for("login"))

        if session.get("role") != "admin":
            flash("Admin access required.")
            return redirect(url_for("dashboard"))

        return function(*args, **kwargs)

    return wrapper


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# REGISTER PAGE
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("Please fill all fields.")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must contain at least 6 characters.")
            return redirect(url_for("register"))

        conn = get_db()
        cursor = conn.cursor()

        try:

            hashed_password = generate_password_hash(password)

            cursor.execute("""
                INSERT INTO users (username, email, password, role)
                VALUES (?, ?, ?, ?)
            """, (
                username,
                email,
                hashed_password,
                "user"
            ))

            conn.commit()

            flash("Registration successful. Please login.")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            flash("Username or email already exists.")
            return redirect(url_for("register"))

        finally:
            conn.close()

    return render_template("register.html")


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        )

        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):

            session.clear()

            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]

            if user["role"] == "admin":
                return redirect(url_for("admin"))

            return redirect(url_for("dashboard"))

        flash("Invalid username or password.")

    return render_template("login.html")


# --------------------------------------------------
# USER DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
    """, (session["user_id"],))

    history = cursor.fetchall()

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE user_id = ?
    """, (session["user_id"],))

    prediction_count = cursor.fetchone()["total"]

    conn.close()

    return render_template(
        "dashboard.html",
        history=history,
        prediction_count=prediction_count,
        training_rows=training_rows,
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2
    )


# --------------------------------------------------
# MOVIE RATING PREDICTION
# --------------------------------------------------

@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():

    if request.method == "POST":

        try:

            if model is None:
                flash("Machine Learning model is not available.")
                return redirect(url_for("prediction"))

            genre = request.form.get("genre", "").strip()

            runtime = float(request.form.get("runtime", 0))
            budget = float(request.form.get("budget", 0))
            votes = int(request.form.get("votes", 0))
            previous_rating = float(
                request.form.get("previous_rating", 0)
            )

            if not genre:
                flash("Please select a genre.")
                return redirect(url_for("prediction"))

            if runtime <= 0 or budget < 0 or votes < 0:
                flash("Please enter valid movie details.")
                return redirect(url_for("prediction"))

            if not 0 <= previous_rating <= 10:
                flash("Previous rating must be between 0 and 10.")
                return redirect(url_for("prediction"))

            input_data = pd.DataFrame([{
                "Genre": genre,
                "Runtime_Min": runtime,
                "Budget_Million": budget,
                "Number_of_Votes": votes,
                "Previous_Rating": previous_rating
            }])

            predicted_rating = float(model.predict(input_data)[0])

            predicted_rating = max(
                0,
                min(10, predicted_rating)
            )

            conn = get_db()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO predictions (
                    user_id,
                    genre,
                    runtime,
                    budget,
                    votes,
                    previous_rating,
                    predicted_rating
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session["user_id"],
                genre,
                runtime,
                budget,
                votes,
                previous_rating,
                predicted_rating
            ))

            conn.commit()
            conn.close()

            return render_template(
                "result.html",
                prediction=predicted_rating,
                genre=genre,
                runtime=runtime,
                budget=budget,
                votes=votes,
                previous_rating=previous_rating
            )

        except (ValueError, TypeError):
            flash("Please enter valid numeric values.")
            return redirect(url_for("prediction"))

        except Exception as error:
            print("Prediction error:", error)
            flash("Prediction failed. Please try again.")
            return redirect(url_for("prediction"))

    return render_template("prediction.html")


# --------------------------------------------------
# PROFILE PAGE
# --------------------------------------------------

@app.route("/profile")
@login_required
def profile():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE id = ?",
        (session["user_id"],)
    )

    user = cursor.fetchone()

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE user_id = ?
    """, (session["user_id"],))

    prediction_count = cursor.fetchone()["total"]

    conn.close()

    return render_template(
        "profile.html",
        user=user,
        prediction_count=prediction_count
    )


# --------------------------------------------------
# ADMIN DASHBOARD
# --------------------------------------------------

@app.route("/admin")
@admin_required
def admin():

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, username, email, role
        FROM users
        ORDER BY id DESC
    """)

    users = cursor.fetchall()

    cursor.execute("""
        SELECT
            predictions.*,
            users.username
        FROM predictions
        LEFT JOIN users
        ON predictions.user_id = users.id
        ORDER BY predictions.id DESC
    """)

    predictions = cursor.fetchall()

    total_users = len(users)

    cursor.execute("SELECT COUNT(*) AS total FROM predictions")
    total_predictions = cursor.fetchone()["total"]

    conn.close()

    return render_template(
        "admin.html",
        users=users,
        predictions=predictions,
        total_users=total_users,
        total_predictions=total_predictions
    )


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out.")

    return redirect(url_for("index"))


# --------------------------------------------------
# INITIALIZE APP FOR RENDER AND LOCAL RUN
# --------------------------------------------------

create_tables()
train_model()


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
