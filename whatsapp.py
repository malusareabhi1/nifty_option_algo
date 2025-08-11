import streamlit as st
from twilio.rest import Client

# Put your Twilio credentials here (or better, use Streamlit secrets)
account_sid = 'AC0ab7d53d5c46e69e2e766441b6ba7de1'
auth_token = 'YOUR_AUTH_TOKEN'  # Replace with your real Auth Token

client = Client(account_sid, auth_token)

st.title("Send WhatsApp Message via Twilio")

to_number = st.text_input("Recipient WhatsApp Number (with country code, e.g. +9198xxxxxx)", value="+919881999644")
message_body = st.text_area("Message Text", value="Hello! This is a test message from Streamlit and Twilio.")

if st.button("Send WhatsApp Message"):
    if to_number and message_body:
        try:
            message = client.messages.create(
                from_='whatsapp:+14155238886',  # Twilio Sandbox WhatsApp number
                body=message_body,
                to=f'whatsapp:{to_number}'
            )
            st.success(f"Message sent! SID: {message.sid}")
        except Exception as e:
            st.error(f"Failed to send message: {e}")
    else:
        st.warning("Please enter both the recipient number and the message.")
