from twilio.rest import Client

# Your Twilio Account SID and Auth Token
#account_sid = "AC0ab7d53d5c46e69e2e766441b6ba7de1"
#auth_token = "91a4377397a2a6341a72a257bf597b1b"
import os
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")

# Initialize the Twilio client
client = Client(account_sid, auth_token)

# Send a WhatsApp message
message = client.messages.create(
    from_="whatsapp:+14155238886",  # Twilio Sandbox WhatsApp number
    body="Hello! This is a test message from Twilio WhatsApp API.",
    to="whatsapp:+91XXXXXXXXXX"  # Replace with your WhatsApp number
)

print(f"Message sent! SID: {message.sid}")
