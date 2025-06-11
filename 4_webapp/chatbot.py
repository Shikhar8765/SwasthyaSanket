# chatbot.py — Gemini-powered Hindi Health Assistant

import google.generativeai as genai

# Make sure your environment variable is set or replace with API key here
genai.configure(api_key="AIzaSyDvbqG_raiWkWdyZd08I862ckD3n7sG8gQ")

def get_bot_response(user_input):
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(user_input)
        return response.text.strip()
    except Exception as e:
        return f"बॉट त्रुटि: {str(e)}"
