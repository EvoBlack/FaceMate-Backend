#!/usr/bin/env python3
"""
Quick script to update existing active sessions with proper radius
Run this to fix the "Session 00:08" that's showing "Too Far"
"""

import mysql.connector
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "face_recognizer"),
}

def update_sessions():
    """Update all active sessions to use proper radius"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # Get current active sessions
        cursor.execute("""
            SELECT id, session_name, radius_meters, is_active 
            FROM attendance_sessions 
            WHERE is_active = TRUE
        """)
        
        sessions = cursor.fetchall()
        
        if not sessions:
            print("No active sessions found.")
            return
        
        print(f"Found {len(sessions)} active session(s):")
        for session in sessions:
            print(f"  - Session ID {session['id']}: {session['session_name']} (radius: {session['radius_meters']}m)")
        
        # Update all active sessions to 20m radius
        # (The backend will add 40m GPS tolerance automatically)
        cursor.execute("""
            UPDATE attendance_sessions 
            SET radius_meters = 20 
            WHERE is_active = TRUE
        """)
        
        updated_count = cursor.rowcount
        conn.commit()
        
        print(f"\n✓ Updated {updated_count} session(s) to 20m radius")
        print("  Backend will add 40m GPS tolerance automatically")
        print("  Effective range: 20m + 40m = 60m")
        
        cursor.close()
        conn.close()
        
        print("\n✓ Done! Please restart the Flutter app (press 'R' in terminal)")
        
    except mysql.connector.Error as e:
        print(f"✗ Database error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Updating Active Session Radius")
    print("=" * 60)
    update_sessions()
