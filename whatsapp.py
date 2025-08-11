from twilio.rest import Client

account_sid = 'AC0ab7d53d5c46e69e2e766441b6ba7de1'
auth_token = '[AuthToken]'
client = Client(account_sid, auth_token)

message = client.messages.create(
  from_='whatsapp:+14155238886',
  content_sid='HXb5b62575e6e4ff6129ad7c8efe1f983e',
  content_variables='{"1":"12/1","2":"3pm"}',
  to='whatsapp:+919881999644'
)

print(message.sid)
