import requests
import json
import os

TOKEN = "8456948452:AAEhL59fX-fKLcMMyfh9iYQCNK_5bCos9Mo"
URL = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

def setup_telegram():
    print("Checking for messages to your bot...")
    response = requests.get(URL).json()
    
    if "result" in response and len(response["result"]) > 0:
        # Get the chat ID from the latest message
        chat_id = response["result"][-1]["message"]["chat"]["id"]
        first_name = response["result"][-1]["message"]["chat"].get("first_name", "User")
        
        print(f"Success! Found Chat ID: {chat_id} for {first_name}")
        
        # Save to environment file
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        with open(env_path, "w") as f:
            f.write(f"TG_TOKEN={TOKEN}\n")
            f.write(f"TG_CHAT_ID={chat_id}\n")
            
        print(f"Saved credentials to {env_path}")
        
        # Send a welcome message
        send_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": "✅ *GRAVITY LAB*\nConnection successful! The trading system is now linked to your Telegram.",
            "parse_mode": "Markdown"
        }
        requests.post(send_url, json=payload)
        print("Test message sent to your Telegram.")
    else:
        print("ERROR: I couldn't find any messages.")
        print("Please open Telegram, search for your new bot, and send it the message: /start")

if __name__ == "__main__":
    setup_telegram()
