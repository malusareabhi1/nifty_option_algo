import streamlit as st
from twilio.rest import Client

# Twilio credentials (store securely in Streamlit secrets)
TWILIO_ACCOUNT_SID = st.secrets["US9fc4f6053ddacf45be89cdd4a161810a"]
TWILIO_AUTH_TOKEN = st.secrets["f57da25825ca02d2f5220c0a9d7cac07"]
TWILIO_WHATSAPP_NUMBER = "whatsapp:+919881999644"  # Twilio sandbox number

def send_whatsapp_message(to_number, message):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message,
            to=f"whatsapp:{to_number}"
        )
        return True
    except Exception as e:
        st.error(f"Error sending message: {e}")
        return False

# Streamlit UI
st.title("📲 Send WhatsApp Message")

recipient_number = st.text_input("Recipient Number (with country code, e.g. +919876543210)")
message_text = st.text_area("Message")

if st.button("Send Message"):
    if recipient_number and message_text:
        if send_whatsapp_message(recipient_number, message_text):
            st.success("✅ Message sent successfully!")
    else:
        st.warning("Please fill in both fields.")
