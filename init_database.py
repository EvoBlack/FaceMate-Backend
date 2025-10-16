#!/usr/bin/env python3
"""
Database initialization script for Face Recognition Attendance System
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
    "charset": "utf8mb4",
    "autocommit": True
}

def create_tables():
    """Create necessary tables for the face recognition system"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("Creating face_encodings table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id INT NOT NULL,
                face_encoding LONGBLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES student(id) ON DELETE CASCADE,
                UNIQUE KEY unique_student (student_id)
            )
        """)
        
        print("Adding recognition_confidence column to attendance table...")
        try:
            cursor.execute("""
                ALTER TABLE attendance 
                ADD COLUMN recognition_confidence FLOAT DEFAULT 0.0
            """)
        except mysql.connector.Error as e:
            if "Duplicate column name" in str(e):
                print("recognition_confidence column already exists")
            else:
                print(f"Error adding recognition_confidence column: {e}")
        
        print("Adding marked_by column to attendance table...")
        try:
            cursor.execute("""
                ALTER TABLE attendance 
                ADD COLUMN marked_by INT,
                ADD FOREIGN KEY (marked_by) REFERENCES teachers(teacher_id)
            """)
        except mysql.connector.Error as e:
            if "Duplicate column name" in str(e):
                print("marked_by column already exists")
            else:
                print(f"Error adding marked_by column: {e}")
        
        print("Creating teacher_subjects junction table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teacher_subjects (
                id INT AUTO_INCREMENT PRIMARY KEY,
                teacher_id INT NOT NULL,
                subject_id INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE CASCADE,
                FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE,
                UNIQUE KEY unique_teacher_subject (teacher_id, subject_id)
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Database initialization completed successfully!")
        
    except mysql.connector.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Initializing database for Face Recognition Attendance System...")
    print("=" * 60)
    create_tables()