from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def welcome():
    return "<h2>Welcome to Math API</h2><p>Use /calculate?operation=add&num1=10&num2=5</p>"

@app.route('/calculate')
def calculate():
    operation = request.args.get('operation')
    num1 = request.args.get('num1', type=float)
    num2 = request.args.get('num2', type=float)

    if not all([operation, num1 is not None, num2 is not None]):
        return jsonify({'error': 'Please provide operation, num1 and num2 parameters'}), 400

    result = None
    if operation == 'add':
        result = num1 + num2
    elif operation == 'subtract':
        result = num1 - num2
    elif operation == 'multiply':
        result = num1 * num2
    elif operation == 'divide':
        if num2 == 0:
            return jsonify({'error': 'Cannot divide by zero'}), 400
        result = num1 / num2
    else:
        return jsonify({'error': 'Invalid operation. Use add, subtract, multiply, divide'}), 400

    return jsonify({
        'operation': operation,
        'num1': num1,
        'num2': num2,
        'result': result
    })

if __name__ == '__main__':
    app.run(debug=True)
