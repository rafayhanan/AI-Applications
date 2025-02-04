import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []
history_string = "\n".join(conversation_history)

def chat(input_text:str):
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

chatbot_ui = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="Simple Chatbot",
    description="Facebook's Blenderbot Model"
)

chatbot_ui.launch()
