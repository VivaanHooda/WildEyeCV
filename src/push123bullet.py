import sys
from pushbullet import Pushbullet

# Add your Pushbullet API key here
PUSHBULLET_API_KEY = ""

def send_push_notification(label):
    try:
        # Initialize Pushbullet
        pb = Pushbullet(PUSHBULLET_API_KEY)
        
        # Create notification
        title = f"{label.upper()} detected"
        message = f"A {label} was detected by your script."
        
        # Send notification
        pb.push_note(title, message)
        print(f"Notification sent: {title} - {message}")
    except Exception as e:
        print(f"Failed to send notification: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No object label provided!")
        sys.exit(1)
    
    # Get the label from the command-line arguments
    object_label = sys.argv[1]
    
    # Send a notification with the label
    send_push_notification(object_label)
