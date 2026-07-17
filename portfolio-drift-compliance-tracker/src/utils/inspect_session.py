import sqlite3
import json
import os

def main():
    session_id = "ec5f9efa-f3a7-49c7-9cb0-cd18841f9041"
    db_path = "src/.adk/session.db"
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM events WHERE session_id = ? ORDER BY id ASC;", (session_id,))
        rows = cursor.fetchall()
        # Find columns
        cursor.execute("PRAGMA table_info(events);")
        columns = [c[1] for c in cursor.fetchall()]
        
        print(f"\n--- Events for Session {session_id} ({len(rows)} events) ---")
        for r in rows:
            event_dict = dict(zip(columns, r))
            print(f"ID: {event_dict.get('id')}")
            
            ev_data_str = event_dict.get('event_data')
            if ev_data_str:
                ev_data = json.loads(ev_data_str)
                print(f"  Author: {ev_data.get('author')}")
                node_info = ev_data.get('node_info') or {}
                print(f"  Node: {node_info.get('path')}")
                print(f"  Output: {ev_data.get('output')}")
                content = ev_data.get('content') or {}
                parts = content.get('parts') or []
                if parts:
                    part = parts[0]
                    text_val = part.get('text')
                    if text_val:
                        print(f"  Content: {text_val[:150]}...")
                    else:
                        print(f"  Content part: {part}")
                else:
                    print(f"  Content: {content}")
            print("-" * 60)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
