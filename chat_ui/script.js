document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    function sendMessage() {
        const messageText = userInput.value.trim();
        if (messageText === '') {
            return; // Don't send empty messages
        }

        // Display user message
        displayMessage(messageText, 'user');

        // Clear input
        userInput.value = '';

        // Send message to backend
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: messageText }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.response) {
                displayMessage(data.response, 'agent');
            } else if (data.error) {
                console.error('Agent error:', data.error);
                displayMessage('Error: Could not get response from agent.', 'agent');
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            displayMessage('Error: Could not connect to the server.', 'agent');
        });
    }

    function displayMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.innerHTML = text.replace(/\n/g, '<br>');
        chatBox.appendChild(messageElement);

        // Scroll to the bottom of the chat box
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Initial message from the agent (optional)
    // displayMessage('Hello! How can I help you today?', 'agent');
});