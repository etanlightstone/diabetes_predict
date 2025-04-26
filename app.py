from flask import Flask, request, jsonify, send_from_directory
from chat_agent import get_agent_response # Import the async function
import os
import asyncio # Import asyncio

app = Flask(__name__)

# We will use the agent instance and MCP server setup from chat_agent.py
# No need to initialize agent here anymore.

@app.route('/')
def serve_index():
    return send_from_directory('chat_ui', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('chat_ui', path)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Get response from the chat agent using the async function
    try:
        agent_response = asyncio.run(get_agent_response(user_message))
        return jsonify({'response': agent_response})
    except Exception as e:
        print(f"Error getting agent response: {e}")
        return jsonify({'error': 'Error getting agent response'}), 500

if __name__ == '__main__':
    # Ensure the chat_ui directory exists
    if not os.path.exists('chat_ui'):
        print("Error: 'chat_ui' directory not found. Please ensure the frontend files are in a 'chat_ui' subdirectory.")
    else:
        # In a production environment, you would use a more robust server like Gunicorn or uWSGI
        # Note: Running Flask with debug=True and asyncio.run can have unexpected behavior.
        # For production, use a proper ASGI server like uvicorn with Flask or a different framework.
        app.run(debug=True)