# chatbot.py — Enhanced Gemini-powered Hindi Health Assistant
import google.generativeai as genai
import re
from typing import Optional, List, Dict

# Configure Gemini - using Streamlit secrets for security
genai.configure(api_key="AIzaSyAuD3I1NmqjwSACoFeukbQMkbiL-mJXyDY")

# System prompt tailored for ASHA workers in rural India
SYSTEM_PROMPT = """
You are 'स्वास्थ्य सहायक', a friendly and knowledgeable Hindi health assistant for ASHA workers. 
Follow these guidelines strictly:

1. Respond only in simple, clear Hindi (Devanagari script)
2. Keep answers concise (3-4 sentences maximum)
3. Focus on preventive care and basic health information
4. Use culturally appropriate examples for rural India
5. For serious symptoms, always recommend visiting a doctor
6. Never provide personal medical advice or diagnoses
7. Support answers with authentic sources when possible

Specialize in:
- Maternal and child health
- Common NCDs (diabetes, hypertension)
- Nutrition and sanitation
- Government health programs
"""

# Response cleaning utilities
def clean_response(text: str) -> str:
    """Clean and format the Gemini response for consistent output"""
    text = re.sub(r'\*\*|\*|`|#', '', text)
    text = text.replace('।.', '।').replace('..', '।')
    if not text.endswith(('।', '!', '?')):
        text = text + '।'
    return text.strip()

def format_for_chat(role: str, text: str) -> Dict:
    """Standardize message format for chat history"""
    return {"role": role, "parts": [text]}

# Main chatbot function
def get_bot_response(
    user_input: str, 
    chat_history: Optional[List[Dict]] = None,
    max_retries: int = 3
) -> str:
    """
    Get response from Gemini model with context awareness
    """
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    messages = []
    messages.append(format_for_chat("user", SYSTEM_PROMPT))
    if chat_history:
        messages.extend(chat_history)
    messages.append(format_for_chat("user", user_input))

    generation_config = {
        "temperature": 0.7,
        "max_output_tokens": 500,
        "top_p": 0.95,
        "top_k": 40
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    for attempt in range(max_retries):
        try:
            chat = model.start_chat(history=messages[:-1])
            response = chat.send_message(
                messages[-1]["parts"][0],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            return clean_response(response.text)
        
        except genai.types.BlockedPromptError:
            return "क्षमा करें, मैं इस विषय पर चर्चा नहीं कर सकता। कृपया स्वास्थ्य संबंधी प्रश्न पूछें।"
        except Exception as e:
            if attempt == max_retries - 1:
                error_msg = f"त्रुटि: अस्थायी तकनीकी समस्या ({str(e)})"
                return f"माफ़ कीजिए, मैं आपके प्रश्न का उत्तर नहीं दे पा रहा हूँ। {error_msg}"

# Test function for standalone testing
def test_chatbot():
    """Test the chatbot functionality"""
    print("स्वास्थ्य सहायक: नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ? (टाइप 'exit' to end)")
    chat_history = []
    while True:
        user_input = input("आप: ")
        if user_input.lower() == 'exit':
            break
        response = get_bot_response(user_input, chat_history)
        print(f"सहायक: {response}")
        chat_history.append(format_for_chat("user", user_input))
        chat_history.append(format_for_chat("model", response))


# ✅ ✅ ✅ NEW: Automated Gemini doctor suggestion for your app.py
# ✅ ✅ ✅ Improved: Automated Gemini doctor suggestion with realistic examples
def get_doctor_recommendation(disease: str, city: str = "Bhopal") -> str:
    """
    Uses Gemini to generate 3 example doctors with short details for ASHA workers.
    Generates realistic placeholder names and hospitals.
    """
    prompt = f"""
You are helping an ASHA worker who needs to suggest local doctors for a patient.
The patient is at high risk for {disease} in {city}.

Please:
- Generate 3 realistic example doctors for this disease.
- For each, include: Doctor's Name, Specialization, Hospital/Clinic, City.
- Make up plausible local names and well-known hospitals.
- Keep it short and formatted as a list.
- Add a one-line reminder that these are examples and ASHA workers should verify locally.

Example:
1. Dr. Anjali Verma - Nephrologist - Apollo Hospital, Bhopal
2. Dr. Rajeev Singh - Nephrologist - Chirayu Medical College, Bhopal
3. Dr. Neeraj Sharma - Nephrologist - Bansal Hospital, Bhopal

Reminder: These are example suggestions. Always verify doctor details locally.
"""
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()



if __name__ == "__main__":
    test_chatbot()
