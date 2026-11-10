
import os
import sqlite3
import pandas as pd

from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)

from werkzeug.security import generate_password_hash, check_password_hash

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "movie-rating-predictor-secret-key-2026"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "MovieDataSet.csv")
DATABASE_PATH = os.path.join(BASE_DIR, "movie_rating.db")


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

def init_db():
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
            movie_title TEXT,
            genre TEXT,
            year INTEGER,
            duration_mins REAL,
            votes REAL,
            predicted_rating REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # Add missing columns to older databases
    user_columns = [
        row["name"] for row in
        cursor.execute("PRAGMA table_info(users)").fetchall()
    ]

    if "email" not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")

    if "role" not in user_columns:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'"
        )

    prediction_columns = [
        row["name"] for row in
        cursor.execute("PRAGMA table_info(predictions)").fetchall()
    ]

    required_prediction_columns = {
        "movie_title": "TEXT",
        "genre": "TEXT",
        "year": "INTEGER",
        "duration_mins": "REAL",
        "votes": "REAL",
        "predicted_rating": "REAL",
        "created_at": "TIMESTAMP"
    }

    for column, data_type in required_prediction_columns.items():
        if column not in prediction_columns:
            cursor.execute(
                f"ALTER TABLE predictions ADD COLUMN {column} {data_type}"
            )

    # Create default admin only if it does not already exist
    admin = cursor.execute(
        "SELECT id FROM users WHERE username = ?",
        ("admin",)
    ).fetchone()

    if admin is None:
        cursor.execute("""
            INSERT INTO users (username, email, password, role)
            VALUES (?, ?, ?, ?)
        """, (
            "admin",
            "admin@moviepredict.com",
            generate_password_hash("Admin@123"),
            "admin"
        ))

    conn.commit()
    conn.close()


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "Moviedataset.csv not found in project folder."
        )

    df = pd.read_csv(DATASET_PATH)

    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "id",
        "title",
        "genre",
        "year",
        "duration_mins",
        "votes",
        "rating"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Dataset column missing: {column}"
            )

    df["title"] = df["title"].fillna("Untitled").astype(str).str.strip()
    df["genre"] = df["genre"].fillna("Unknown").astype(str).str.strip()

    numeric_columns = [
        "id",
        "year",
        "duration_mins",
        "votes",
        "rating"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "year",
            "duration_mins",
            "votes",
            "rating"
        ]
    )

    df = df[
        (df["rating"] >= 0) &
        (df["rating"] <= 10) &
        (df["duration_mins"] > 0) &
        (df["votes"] >= 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid movie records found in dataset.")

    return df


# --------------------------------------------------
# TRAIN MACHINE LEARNING MODEL
# --------------------------------------------------

FEATURES = [
    "genre",
    "year",
    "duration_mins",
    "votes"
]

NUMERIC_FEATURES = [
    "year",
    "duration_mins",
    "votes"
]

CATEGORICAL_FEATURES = ["genre"]

model = None
movie_df = pd.DataFrame()
model_metrics = {}


def train_model():

    global model, movie_df, model_metrics

    movie_df = load_dataset()

    X = movie_df[FEATURES]
    y = movie_df["rating"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X, y)

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y, predictions)

    model_metrics = {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 3),
        "accuracy": round(max(0, min(100, r2 * 100)), 2)
    }

    print("Model trained successfully!")
    print("Dataset rows:", len(movie_df))
    print("MAE:", model_metrics["mae"])
    print("RMSE:", model_metrics["rmse"])
    print("R2 Score:", model_metrics["r2"])


# --------------------------------------------------
# LOGIN REQUIRED DECORATOR
# --------------------------------------------------

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:
            flash("Please login first.", "error")
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
            flash("Please login first.", "error")
            return redirect(url_for("login"))

        if session.get("role") != "admin":
            flash("Admin access only.", "error")
            return redirect(url_for("dashboard"))

        return function(*args, **kwargs)

    return wrapper


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def index():

    if "user_id" in session:
        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


# --------------------------------------------------
# REGISTER
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("Please fill all fields.", "error")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must contain at least 6 characters.", "error")
            return redirect(url_for("register"))

        conn = get_db()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO users (username, email, password, role)
                VALUES (?, ?, ?, ?)
            """, (
                username,
                email,
                generate_password_hash(password),
                "user"
            ))

            conn.commit()
            flash("Registration successful. Please login.", "success")

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "error")
            return redirect(url_for("register"))

        finally:
            conn.close()

    return render_template("register.html")


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = get_db()

        user = conn.execute("""
            SELECT * FROM users
            WHERE username = ?
        """, (username,)).fetchone()

        conn.close()

        if user:

            stored_password = user["password"]

            password_valid = False

            try:
                password_valid = check_password_hash(
                    stored_password,
                    password
                )
            except (ValueError, TypeError):
                password_valid = False

            # Compatibility with old plain-text passwords
            if not password_valid and stored_password == password:
                password_valid = True

            if password_valid:

                session.clear()

                session["user_id"] = user["id"]
                session["username"] = user["username"]
                session["email"] = user["email"]
                session["role"] = user["role"] or "user"

                flash("Login successful.", "success")

                return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():

    conn = get_db()

    total_users = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    total_predictions = conn.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    user_predictions = conn.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE user_id = ?
    """, (session["user_id"],)).fetchone()[0]

    recent_predictions = conn.execute("""
        SELECT * FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 5
    """, (session["user_id"],)).fetchall()

    conn.close()

    total_movies = len(movie_df) if not movie_df.empty else 0

    average_rating = 0

    if total_movies > 0:
        average_rating = round(
            float(movie_df["rating"].mean()), 2
        )

    return render_template(
        "dashboard.html",
        total_movies=total_movies,
        average_rating=average_rating,
        total_users=total_users,
        total_predictions=total_predictions,
        user_predictions=user_predictions,
        recent_predictions=recent_predictions
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
                raise ValueError("Machine learning model is not available.")

            movie_title = request.form.get(
                "movie_title",
                request.form.get("title", "New Movie")
            ).strip()

            genre = request.form.get("genre", "").strip()

            year = float(request.form.get("year", ""))
            duration_mins = float(
                request.form.get("duration_mins", "")
            )
            votes = float(request.form.get("votes", ""))

            if not genre:
                raise ValueError("Please select a movie genre.")

            if not movie_title:
                movie_title = "New Movie"

            if year < 1888 or year > 2100:
                raise ValueError("Please enter a valid release year.")

            if duration_mins <= 0:
                raise ValueError("Duration must be greater than zero.")

            if votes < 0:
                raise ValueError("Votes cannot be negative.")

            input_data = pd.DataFrame([{
                "genre": genre,
                "year": year,
                "duration_mins": duration_mins,
                "votes": votes
            }])

            predicted_rating = float(
                model.predict(input_data)[0]
            )

            predicted_rating = max(
                0,
                min(10, predicted_rating)
            )

            predicted_rating = round(
                predicted_rating, 2
            )

            conn = get_db()

            conn.execute("""
                INSERT INTO predictions (
                    user_id,
                    movie_title,
                    genre,
                    year,
                    duration_mins,
                    votes,
                    predicted_rating
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session["user_id"],
                movie_title,
                genre,
                int(year),
                duration_mins,
                votes,
                predicted_rating
            ))

            conn.commit()
            conn.close()

            return render_template(
                "prediction.html",
                predicted_rating=predicted_rating,
                result=predicted_rating
            )

        except Exception as error:

            flash(
                "Prediction error: " + str(error),
                "error"
            )

    return render_template("prediction.html")


# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

@app.route("/analytics")
@login_required
def analytics():

    try:

        if movie_df.empty:
            df = load_dataset()
        else:
            df = movie_df.copy()

        df["genre"] = (
            df["genre"].fillna("Unknown").astype(str).str.strip()
        )

        df["title"] = (
            df["title"].fillna("Untitled").astype(str).str.strip()
        )

        df["rating"] = pd.to_numeric(
            df["rating"], errors="coerce"
        )

        df["votes"] = pd.to_numeric(
            df["votes"], errors="coerce"
        )

        df = df.dropna(subset=["rating", "votes"])

        df = df[
            (df["rating"] >= 0) &
            (df["rating"] <= 10) &
            (df["votes"] >= 0)
        ].copy()

        if df.empty:
            raise ValueError("No valid movie records found.")

        total_movies = len(df)

        avg_rating = round(
            float(df["rating"].mean()), 2
        )

        max_rating = round(
            float(df["rating"].max()), 2
        )

        min_rating = round(
            float(df["rating"].min()), 2
        )

        # Rating distribution
        rating_labels = [
            "0-1", "1-2", "2-3", "3-4", "4-5",
            "5-6", "6-7", "7-8", "8-9", "9-10"
        ]

        rating_counts = []

        for i in range(10):

            if i == 9:
                count = (
                    (df["rating"] >= 9) &
                    (df["rating"] <= 10)
                ).sum()

            else:
                count = (
                    (df["rating"] >= i) &
                    (df["rating"] < i + 1)
                ).sum()

            rating_counts.append(int(count))

        # Genre analysis
        genre_data = df.groupby("genre").agg(
            movie_count=("id", "count"),
            average_rating=("rating", "mean")
        ).sort_values(
            "movie_count",
            ascending=False
        )

        genre_labels = genre_data.index.tolist()

        genre_counts = (
            genre_data["movie_count"].astype(int).tolist()
        )

        genre_ratings = [
            round(float(value), 2)
            for value in genre_data["average_rating"].tolist()
        ]

        # Top 10 movies by votes
        votes_sample = df.sort_values(
            by="votes",
            ascending=False
        ).head(10)

        votes_labels = votes_sample["title"].tolist()

        votes_values = (
            votes_sample["votes"].astype(int).tolist()
        )

        return render_template(
            "analytics.html",
            total_movies=total_movies,
            avg_rating=avg_rating,
            max_rating=max_rating,
            min_rating=min_rating,
            rating_labels=rating_labels,
            rating_counts=rating_counts,
            genre_labels=genre_labels,
            genre_counts=genre_counts,
            genre_ratings=genre_ratings,
            votes_labels=votes_labels,
            votes_values=votes_values
        )

    except Exception as error:

        print("ANALYTICS ERROR:", error)

        flash(
            "Unable to load analytics: " + str(error),
            "error"
        )

        return redirect(url_for("dashboard"))


# --------------------------------------------------
# DATASET
# --------------------------------------------------

@app.route("/dataset")
@login_required
def dataset():

    try:

        df = load_dataset()

        selected_genre = request.args.get(
            "genre",
            "All Genres"
        )

        if selected_genre != "All Genres":
            df = df[
                df["genre"].str.lower() ==
                selected_genre.lower()
            ]

        movies = df.to_dict(orient="records")

        genres = sorted(
            movie_df["genre"].dropna().unique().tolist()
        )

        total_movies = len(movie_df)

        return render_template(
            "dataset.html",
            movies=movies,
            genres=genres,
            selected_genre=selected_genre,
            total_movies=total_movies
        )

    except Exception as error:

        print("DATASET ERROR:", error)

        flash(
            "Unable to load dataset: " + str(error),
            "error"
        )

        return redirect(url_for("dashboard"))


# --------------------------------------------------
# PROFILE
# --------------------------------------------------

@app.route("/profile")
@login_required
def profile():

    conn = get_db()

    user = conn.execute("""
        SELECT * FROM users
        WHERE id = ?
    """, (session["user_id"],)).fetchone()

    prediction_count = conn.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE user_id = ?
    """, (session["user_id"],)).fetchone()[0]

    conn.close()

    return render_template(
        "profile.html",
        user=user,
        prediction_count=prediction_count
    )


# --------------------------------------------------
# FEEDBACK
# --------------------------------------------------

@app.route("/feedback", methods=["GET", "POST"])
@login_required
def feedback():

    if request.method == "POST":

        message = request.form.get(
            "message",
            ""
        ).strip()

        if not message:
            flash("Please enter your feedback.", "error")
            return redirect(url_for("feedback"))

        conn = get_db()

        conn.execute("""
            INSERT INTO feedback (user_id, message)
            VALUES (?, ?)
        """, (
            session["user_id"],
            message
        ))

        conn.commit()
        conn.close()

        flash("Thank you for your feedback!", "success")

        return redirect(url_for("feedback"))

    return render_template("feedback.html")


# --------------------------------------------------
# ADMIN DASHBOARD
# --------------------------------------------------

@app.route("/admin")
@login_required
@admin_required
def admin():

    conn = get_db()

    total_users = conn.execute("""
        SELECT COUNT(*) FROM users
    """).fetchone()[0]

    total_predictions = conn.execute("""
        SELECT COUNT(*) FROM predictions
    """).fetchone()[0]

    total_feedback = conn.execute("""
        SELECT COUNT(*) FROM feedback
    """).fetchone()[0]

    users = conn.execute("""
        SELECT id, username, email, role
        FROM users
        ORDER BY id DESC
    """).fetchall()

    predictions = conn.execute("""
        SELECT
            predictions.id,
            predictions.movie_title,
            predictions.genre,
            predictions.predicted_rating,
            predictions.created_at,
            users.username
        FROM predictions
        LEFT JOIN users
        ON predictions.user_id = users.id
        ORDER BY predictions.id DESC
    """).fetchall()

    feedback_list = conn.execute("""
        SELECT
            feedback.id,
            feedback.message,
            feedback.created_at,
            users.username
        FROM feedback
        LEFT JOIN users
        ON feedback.user_id = users.id
        ORDER BY feedback.id DESC
    """).fetchall()

    conn.close()

    return render_template(
        "admin.html",
        total_users=total_users,
        total_predictions=total_predictions,
        total_feedback=total_feedback,
        users=users,
        predictions=predictions,
        feedback_list=feedback_list
    )


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash("You have been logged out.", "success")

    return redirect(url_for("login"))


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

init_db()

try:
    train_model()
except Exception as error:
    print("MODEL TRAINING ERROR:", error)


if __name__ == "__main__":
    app.run(debug=True)
