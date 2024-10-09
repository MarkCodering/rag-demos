import gradio as gr
import requests

# Function to chat with the FastAPI server
def chat_with_fastapi(prompt):
    # Let's talk to the FastAPI server
    url = "http://localhost:8080/generate_response"
    try:
        response = requests.post(url, data={'prompt': prompt})
        if response.status_code == 200:
            result = response.json()
            return result['response']
        else:
            return "Hmm, I couldn't connect to the FastAPI server. Please try again later."
    except requests.exceptions.RequestException as e:
        return f"Oops, something went wrong: {e}"

# Gradio Chatbot Interface
def gradio_chatbot(prompt):
    # Get the response from our FastAPI friend
    response = chat_with_fastapi(prompt)
    return response

# Setting up our Gradio chat interface with message stream
with gr.Blocks() as demo:
    with gr.Row() as msg:
        # Textbox for user input
        msg = gr.Textbox(lines=1, placeholder="Type your message here...", label="User")
        # Button to clear the chat history
        clear = gr.Button("Clear Chat")
        
    # Textbox to display the chat history
    output = gr.Textbox(lines=10, placeholder="Chat history will appear here...", label="Chatbot")

    def submit_message(user_message):
        # Get chatbot response
        response = gradio_chatbot(user_message)
        # Append chatbot response to chat history

        return response

    # When you hit enter, let's send the message and update the chat history
    msg.submit(submit_message, inputs=[msg], outputs=[output])
    # Clear the chat when you click the button
    clear.click(lambda: ("", []), None, [output], queue=False)

# Time to launch our chat!
demo.launch()