import json
from datetime import datetime

def log_chat_history(session_title, chat_history):
    """Log chat history to a JSON file."""
    log_file = f"chat_logs/{session_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)
    print(f"Chat history logged to {log_file}.")
