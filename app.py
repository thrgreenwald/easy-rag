from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# TODO: IN PROGRESS
# Endpoint to answer a user question
@app.route('/question', methods=['POST'])
def answer_user_question():
    try:
        # Ensure the request contains JSON data
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({'error': 'Invalid request. Please provide conversation messages.'}), 400

        question = data['question']

        return jsonify({'answer': question}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8000)
