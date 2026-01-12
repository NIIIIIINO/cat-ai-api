from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "OK"

@app.route("/predict", methods=["POST"])
def predict():
    return jsonify({
        "status": "ok",
        "message": "AI server is ready"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
