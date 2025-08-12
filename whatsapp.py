from twilio.rest import Client

# Your Twilio Account SID and Auth Token
account_sid = "AC0ab7d53d5c46e69e2e766441b6ba7de1"
auth_token = "AC0ab7d53d5c46e69e2e766441b6ba7de1"

# Initialize the Twilio client
client = Client(account_sid, auth_token)

# Send a WhatsApp message
message = client.messages.create(
    from_="whatsapp:+14155238886",  # Twilio Sandbox WhatsApp number
    body="Hello! This is a test message from Twilio WhatsApp API.",
    to="whatsapp:+91XXXXXXXXXX"  # Replace with your WhatsApp number
)

print(f"Message sent! SID: {message.sid}")
