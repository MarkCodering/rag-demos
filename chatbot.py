import gradio as gr
import requests

# Constants
FASTAPI_URL = "http://localhost:8080/generate_response"

# Function to chat with the FastAPI server
def chat_with_fastapi(prompt):
    """Send a prompt to the FastAPI server and return the response."""
    try:
        response = requests.post(FASTAPI_URL, data={'prompt': prompt})
        response.raise_for_status()  # Raise an error for bad responses
        result = response.json()
        return result.get('response', "No response found.")
    except requests.exceptions.RequestException as e:
        return f"Oops, something went wrong: {e}"

# Gradio Chatbot Interface
def gradio_chatbot(prompt):
    """Get the response from the FastAPI server for a given prompt."""
    return chat_with_fastapi(prompt)

# Setting up the Gradio chat interface
with gr.Blocks() as demo:
    with gr.Row():
        msg = gr.Textbox(lines=1, placeholder="Type your message here...", label="User")
        clear = gr.Button("Clear Chat")

    output = gr.Textbox(lines=2, placeholder="Chatbot response will appear here...", label="Chatbot", interactive=False)

    # Bind the submit function to the message input
    msg.submit(lambda user_message: gradio_chatbot(user_message), inputs=msg, outputs=output)
    # Clear the chat when the button is clicked
    clear.click(lambda: "", None, output, queue=False)

# Launch the Gradio app
demo.launch()