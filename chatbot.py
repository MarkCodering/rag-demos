import gradio as gr
import requests

# Function to send prompt to FastAPI server and get the response
def chat_with_fastapi(prompt):
    # Send a POST request to FastAPI server
    url = "http://localhost:8080/generate_response"
    response = requests.post(url, data={'prompt': prompt})
    
    if response.status_code == 200:
        result = response.json()
        return result['response']
    else:
        return "Error: Unable to connect to FastAPI server."

# Create a Gradio Chat Interface
def gradio_chatbot(prompt):
    response = chat_with_fastapi(prompt)
    return response

# Define Gradio Interface
with gr.Blocks() as demo:
    msg = gr.Textbox(placeholder="Enter your prompt here...", label="Prompt")
    output = gr.Textbox(label="Model Output")
    clear = gr.Button("Clear")

    def submit_message(user_message):
        response = gradio_chatbot(user_message)
        return response

    msg.submit(submit_message, inputs=msg, outputs=output)
    clear.click(lambda: "", None, output, queue=False)

# Launch the Gradio app
demo.launch()
