"""
ingest.py — Parse the CSV and return a flat chronological list of messages.

Each message is a dict:
    {
        "msg_id":      int,   # global 0-based index across all conversations
        "conv_id":     int,   # which conversation row (0-based)
        "speaker":     str,   # "User 1" or "User 2"
        "text":        str,   # raw message text
    }
"""

import csv
import re
from pathlib import Path


def load_messages(csv_path: str) -> list[dict]:
    """
    Read the CSV and flatten all conversations into one chronological
    list of messages, preserving the row order as the 'day' order.
    """
    messages = []
    msg_id = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for conv_id, row in enumerate(reader):
            if not row:
                continue
            conversation_text = row[0]
            lines = conversation_text.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Match "User 1: ..." or "User 2: ..."
                m = re.match(r"^(User\s*\d+)\s*:\s*(.+)$", line)
                if m:
                    speaker = m.group(1).strip()
                    text = m.group(2).strip()
                    messages.append({
                        "msg_id": msg_id,
                        "conv_id": conv_id,
                        "speaker": speaker,
                        "text": text,
                    })
                    msg_id += 1

    return messages


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/conversations.csv"
    msgs = load_messages(path)
    print(f"Loaded {len(msgs)} messages from {path}")
    for m in msgs[:5]:
        print(m)
