from flask import Flask, jsonify
from flask import request


app = Flask(__name__)

# Define a route for the root URI
@app.route('/')
def home():
    return "Welcome to the Flask API!"

# Define a route for a specific URI
@app.route('/api/recent_data', methods=['GET'])
def get_data():
    data = {
        "name": "John Doe",
        "age": 30,
        "city": "Budapest"
    }
    return jsonify(data)

# Define another route for a different URI
@app.route('/api/historic_data', methods=['GET'])
def get_message():
    d = request.args.get('date')
    print(d)

    return jsonify({"date": str(d)})

if __name__ == '__main__':
    app.run(debug=True)