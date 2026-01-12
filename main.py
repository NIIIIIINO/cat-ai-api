from flask import Flask

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "OK"

@app.route("/ping", methods=["GET"])
def ping():
    return "pong"
