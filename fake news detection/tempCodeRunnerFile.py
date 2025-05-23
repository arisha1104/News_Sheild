from flask import Flask, request, redirect, url_for, render_template, session, jsonify
import pickle
import pandas as pd
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used for sessions

# Hardcoded user credentials (you can modify these)
USER_CREDENTIALS = {
    "username": "admin",
    "password": "password123"
}

# Load the trained model and vectorizer
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the dataset
df = pd.read_csv("cleaned_news.csv")

# Your News API Key and URL
NEWS_API_KEY = "your_news_api_key_here"
NEWS_API_URL = "https://newsapi.org/v2/everything"


@app.route("/")
def index():
    # Redirect to login page if not logged in
    if "logged_in" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("home"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        if USER_CREDENTIALS["username"] == username and USER_CREDENTIALS["password"] == password:
            session["logged_in"] = True
            return redirect(url_for("home"))
        else:
            return "Invalid credentials. Please try again."

    return render_template("login.html")


@app.route("/home")
def home():
    if "logged_in" not in session:
        return redirect(url_for("login"))
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news_text"].strip()
        
        # Find exact match first
        exact_match = df[(df["title"].str.lower() == news_text.lower()) | (df["text"].str.lower() == news_text.lower())]

        if not exact_match.empty:
            exact_match = exact_match.copy()
            exact_match["label"] = exact_match["label"].apply(lambda x: "Real News" if x == 1 else "Fake News")
            
            prediction = exact_match.iloc[0]["label"]
            return jsonify({"prediction": prediction, "matched_articles": exact_match[["title", "text", "label"]].to_dict(orient="records")})
        
        # Otherwise, find partial matches (suggested articles)
        matched_articles = df[df["text"].str.contains(news_text, case=False, na=False) | df["title"].str.contains(news_text, case=False, na=False)]
        
        if not matched_articles.empty:
            matched_articles = matched_articles.copy()
            matched_articles["label"] = matched_articles["label"].apply(lambda x: "Real News" if x == 1 else "Fake News")

            return jsonify({"prediction": "See matched articles", "matched_articles": matched_articles[["title", "text", "label"]].to_dict(orient="records")})

        else:
            return jsonify({"prediction": "Not matched in dataset", "matched_articles": []})

    return render_template("predict_form.html")


@app.route("/real_time_search", methods=["GET", "POST"])
def real_time_search():
    if request.method == "POST":
        query = request.form["query"]
        response = requests.get(NEWS_API_URL, params={
            "q": query,
            "apiKey": NEWS_API_KEY
        })

        articles = response.json().get("articles", [])
        return render_template("real_time_results.html", articles=articles)

    return render_template("real_time_search.html")


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
