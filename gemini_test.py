import google.generativeai as genai

genai.configure(api_key="AIzaSyDvbqG_raiWkWdyZd08I862ckD3n7sG8gQ")

model = genai.GenerativeModel("models/gemini-1.5-flash")
response = model.generate_content("भारत में मधुमेह के लक्षण क्या हैं?")
print("Bot:", response.text)
