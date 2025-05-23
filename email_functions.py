from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import os.path
import base64

# Define the scope for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

def authenticate_gmail():
    """Authenticate the Gmail API and return the service."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

def search_messages(service, query, user_id='me'):
    """Search for specific messages in Gmail."""
    results = service.users().messages().list(userId=user_id, q=query).execute()
    messages = results.get('messages', [])
    return messages

def get_message_content(service, user_id, msg_id):
    """Retrieve the full content of a Gmail message."""
    message = service.users().messages().get(userId=user_id, id=msg_id, format='full').execute()
    payload = message['payload']
    body = ""
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                body = base64.urlsafe_b64decode(part['body']['data']).decode()
                break
            elif part['mimeType'] == 'text/html':
                body = base64.urlsafe_b64decode(part['body']['data']).decode()
                break
    elif 'body' in payload and 'data' in payload['body']:
        body = base64.urlsafe_b64decode(payload['body']['data']).decode()
    return body

def analyze_email_content(content):
    """Analyze the email content to determine the return value."""
    passport_acceptance_str = "The passport meets the criteria for approval"
    photo_acceptance_str = "The person's photo meets the criteria for approval"
    if passport_acceptance_str in content and photo_acceptance_str in content:
        return 1
    elif passport_acceptance_str in content:
        return 2
    elif photo_acceptance_str in content:
        return 3
    else:
        return 4

def mark_as_read(service, msg_id, user_id='me'):
    """Mark an email as read."""
    try:
        result = service.users().messages().modify(
            userId=user_id,
            id=msg_id, 
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        # if 'labelIds' in result and 'UNREAD' not in result['labelIds']:
        #     print(f"Message {msg_id} marked as read successfully.")
        # else:
        #     print(f"Failed to mark message {msg_id} as read.")
    except Exception as e:
        print(f"Error marking message {msg_id} as read: {e}")


def read_last_unread_email_with_passport(passportNumber):
    #1: verified document
    #2: passport accepted photo rejected
    #3: passport rejected photo accepted
    #4: passport rejected photo rejected
    #5: no email found
    """
    Read the most recent unread email from noreply@mohre.gov.ae with subject 'Rejected Document' 
    and containing the specified passportNumber.
    """
    # Authenticate Gmail service
    service = authenticate_gmail()
    query = f'is:unread from:noreply@mohre.gov.ae subject:"Verified Document - {passportNumber}"'
    messages = search_messages(service, query)
    if messages:
        last_message = messages[-1]  # Last message in the list
        msg_id = last_message['id']
        mark_as_read(service, msg_id)
        return 1


    query = 'is:unread from:noreply@mohre.gov.ae subject:"Rejected Document"'
    messages = search_messages(service, query)

    if not messages:
        # print(f"No matching emails found: {passportNumber}")
        return 5

    # Iterate through messages in reverse order (most recent first)
    for message in reversed(messages):
        msg_id = message['id']

        # Retrieve the content of the message
        content = get_message_content(service, 'me', msg_id)
        # Check if the passportNumber exists in the content
        if passportNumber in content:
            # Analyze the content
            result = analyze_email_content(content)
            # print(f"Message ID: {msg_id}, Result: {result}")

            # Mark the message as read
            mark_as_read(service, msg_id)

            return result

    # print(f"No unread emails contain the passport number: {passportNumber}.")
    return 5

# Call the function

