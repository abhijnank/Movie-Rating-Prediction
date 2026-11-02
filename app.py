from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os
import pandas as pd
import numpy as np

from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)
app.secret_key = "movie_rating_prediction_secret_key"

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
# TRAIN LINEAR REGRESSION MODEL
# --------------------------------------------------

def train_model():
    global model, mae, mse, rmse, r2, training_rows

    if not os.path.exists(DATASET):
        raise FileNotFoundError(
            "rating.csv not found. Keep rating.csv in the same folder as app.py."
        )

    df = pd.read_csv(DATASET)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in rating.csv: {missing_columns}"
        )

    # Clean Genre values
    df["Genre"] = df["Genre"].astype("string").str.strip()

    # Convert numeric columns
    for column in NUMERIC_FEATURES + [TARGET]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(subset=required_columns)

    df = df[df["Genre"] != ""]

    if len(df) < 5:
        raise ValueError(
            "Dataset must contain at least 5 valid rows."
        )

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Encode Genre and pass numeric features
    preprocessor = ColumnTransformer(
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

    # Linear Regression Pipeline
    evaluation_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
    )

    evaluation_model.fit(X_train, y_train)

    predictions = evaluation_model.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    mse = float(mean_squared_error(y_test, predictions))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, predictions))

    # Train final model using complete dataset
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
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
            ),
            ("regressor", LinearRegression())
        ]
    )

    model.fit(X, y)

    training_rows = len(df)

    print("Model trained successfully!")
    print("Algorithm: Linear Regression")
    print("Genre included in training.")
    print("Dataset rows:", training_rows)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_db():
    con = sqlite3.connect(DATABASE)
    con.row_factory = sqlite3.Row
    return con


# --------------------------------------------------
# CREATE DATABASE TABLES
# --------------------------------------------------

def create_tables():

    con = get_db()
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            genre TEXT,
            runtime REAL,
            budget REAL,
            votes REAL,
            previous_rating REAL,
            predicted_rating REAL NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Update older database if columns are missing
    columns = [
        row["name"]
        for row in cur.execute(
            "PRAGMA table_info(predictions)"
        ).fetchall()
    ]

    required_db_columns = {
        "user_id": "INTEGER",
        "genre": "TEXT",
        "runtime": "REAL",
        "budget": "REAL",
        "votes": "REAL",
        "previous_rating": "REAL",
        "predicted_rating": "REAL NOT NULL DEFAULT 0",
        "created_at": "TIMESTAMP"
    }

    for column_name, column_type in required_db_columns.items():

        if column_name not in columns:
            cur.execute(
                f"ALTER TABLE predictions ADD COLUMN "
                f"{column_name} {column_type}"
            )

    # Create default admin account
    admin = cur.execute(
        "SELECT id FROM users WHERE username = ?",
        ("admin",)
    ).fetchone()

    if admin is None:

        hashed_password = generate_password_hash("Admin@123")

        cur.execute("""
            INSERT INTO users (username, email, password, role)
            VALUES (?, ?, ?, ?)
        """, (
            "admin",
            "admin@movierating.com",
            hashed_password,
            "admin"
        ))

    con.commit()
    con.close()

    print("Database tables created successfully!")


# --------------------------------------------------
# LOGIN REQUIRED
# --------------------------------------------------

def login_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:
            flash("Please login first.", "error")
            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return decorated_function


# --------------------------------------------------
# ADMIN REQUIRED
# --------------------------------------------------

def admin_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:
            flash("Please login first.", "error")
            return redirect(url_for("login"))

        if session.get("role") != "admin":
            flash("Access denied. Admin only.", "error")
            return redirect(url_for("dashboard"))

        return function(*args, **kwargs)

    return decorated_function


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
            flash("Please fill all required fields.", "error")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        con = get_db()

        try:

            con.execute("""
                INSERT INTO users
                (username, email, password, role)
                VALUES (?, ?, ?, ?)
            """, (
                username,
                email,
                hashed_password,
                "user"
            ))

            con.commit()

            flash("Registration successful. Please login.", "success")

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            flash("Username or email already exists.", "error")

        finally:
            con.close()

    return render_template("register.html")


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        con = get_db()

        user = con.execute("""
            SELECT * FROM users
            WHERE username = ?
        """, (username,)).fetchone()

        con.close()

        if user and check_password_hash(user["password"], password):

            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]

            flash("Login successful.", "success")

            if user["role"] == "admin":
                return redirect(url_for("admin"))

            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():

    con = get_db()

    history = con.execute("""
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
    """, (session["user_id"],)).fetchall()

    prediction_count = con.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE user_id = ?
    """, (session["user_id"],)).fetchone()["total"]

    con.close()

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

            genre = request.form.get("genre", "").strip()

            runtime = float(request.form.get("runtime", ""))
            budget = float(request.form.get("budget", ""))
            votes = float(request.form.get("votes", ""))
            previous_rating = float(
                request.form.get("previous_rating", "")
            )

            if not genre:
                raise ValueError("Please select a movie genre.")

            if not all(np.isfinite([
                runtime,
                budget,
                votes,
                previous_rating
            ])):
                raise ValueError("Enter valid numeric values.")

            if runtime <= 0:
                raise ValueError("Runtime must be greater than zero.")

            if budget < 0:
                raise ValueError("Budget cannot be negative.")

            if votes < 0:
                raise ValueError("Number of votes cannot be negative.")

            if not 0 <= previous_rating <= 10:
                raise ValueError(
                    "Previous rating must be between 0 and 10."
                )

            if model is None:
                raise RuntimeError("Machine learning model is not trained.")

            input_data = pd.DataFrame([{
                "Genre": genre,
                "Runtime_Min": runtime,
                "Budget_Million": budget,
                "Number_of_Votes": votes,
                "Previous_Rating": previous_rating
            }])

            predicted_rating = float(
                model.predict(input_data)[0]
            )

            predicted_rating = max(
                0.0,
                min(10.0, predicted_rating)
            )

            # Save prediction in database
            con = get_db()

            con.execute("""
                INSERT INTO predictions
                (
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

            con.commit()
            con.close()

            return render_template(
                "result.html",
                prediction=round(predicted_rating, 2),
                genre=genre,
                runtime=runtime,
                budget=budget,
                votes=votes,
                previous_rating=previous_rating
            )

        except (ValueError, TypeError) as error:

            flash(str(error), "error")
            return redirect(url_for("prediction"))

        except Exception as error:

            print("Prediction error:", error)

            flash(
                "Prediction failed. Check the Flask terminal for the error.",
                "error"
            )

            return redirect(url_for("prediction"))

    return render_template("prediction.html")


# --------------------------------------------------
# PROFILE PAGE
# --------------------------------------------------

@app.route("/profile")
@login_required
def profile():

    con = get_db()

    user = con.execute("""
        SELECT id, username, email, role
        FROM users
        WHERE id = ?
    """, (session["user_id"],)).fetchone()

    prediction_count = con.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE user_id = ?
    """, (session["user_id"],)).fetchone()["total"]

    con.close()

    return render_template(
        "profile.html",
        user=user,
        prediction_count=prediction_count
    )


# --------------------------------------------------
# ADMIN PAGE
# --------------------------------------------------

@app.route("/admin")
@admin_required
def admin():

    con = get_db()

    users = con.execute("""
        SELECT id, username, email, role
        FROM users
        ORDER BY id DESC
    """).fetchall()

    predictions = con.execute("""
        SELECT
            predictions.*,
            users.username
        FROM predictions
        LEFT JOIN users
        ON users.id = predictions.user_id
        ORDER BY predictions.id DESC
    """).fetchall()

    total_users = con.execute("""
        SELECT COUNT(*) AS total
        FROM users
        WHERE role != 'admin'
    """).fetchone()["total"]

    total_predictions = con.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
    """).fetchone()["total"]

    con.close()

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

    flash("You have been logged out.", "success")

    return redirect(url_for("index"))


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    create_tables()

    train_model()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )