from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import bcrypt
import os
import sys
import pickle
import numpy as np
import cv2
from datetime import datetime, date
import json
import base64
try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from facenet_pytorch import MTCNN, InceptionResnetV1
    from PIL import Image
    print("PyTorch and FaceNet-PyTorch imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install torch torchvision facenet-pytorch")
    sys.exit(1)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, 
     origins="*", 
     allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"], 
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     supports_credentials=True)

# Global variables for models
mtcnn = None
facenet = None
device = None

def initialize_models():
    """Initialize face detection and recognition models"""
    global mtcnn, facenet, device
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Initialize MTCNN for face detection with optimized settings
        print("Initializing MTCNN...")
        mtcnn = MTCNN(
            image_size=160, 
            margin=20,  # Increased margin for better face context
            min_face_size=30,  # Lowered to detect smaller/distant faces (was 40)
            thresholds=[0.6, 0.7, 0.7],  # More lenient thresholds for better detection (was 0.7, 0.8, 0.8)
            factor=0.709, 
            post_process=True,
            keep_all=False,  # Only keep the best face
            device=device
        )
        print("MTCNN initialized successfully with optimized settings")

        # Initialize FaceNet for face recognition
        print("Initializing FaceNet...")
        facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        print("FaceNet initialized successfully")
        
        print("All face recognition models initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

# Initialize models at startup
models_initialized = initialize_models()

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "face_recognizer"),
    "charset": "utf8mb4",
    "autocommit": True
}

def get_db_connection():
    """Get database connection"""
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        return None

def extract_face_embedding(image):
    """Extract face embedding using MTCNN and FaceNet with quality checks"""
    global mtcnn, facenet, device
    
    if not models_initialized or mtcnn is None or facenet is None:
        print("Models not initialized")
        return None
    
    try:
        # Convert to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            # Check image quality
            if image.size == 0:
                print("Empty image provided")
                return None
            
            # Check if image is too small
            if image.shape[0] < 100 or image.shape[1] < 100:
                print(f"Image too small: {image.shape}")
                return None
            
            # Check brightness (avoid too dark or too bright images)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            brightness = np.mean(gray)
            if brightness < 30 or brightness > 225:
                print(f"Poor lighting detected: brightness = {brightness:.1f}")
                # Continue but log warning
            
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)
        
        print(f"Processing image of size: {image.size}")
        
        # Extract face using MTCNN (this detects, crops, and preprocesses the face)
        face_tensor = mtcnn(image)
        
        if face_tensor is None:
            print("No face detected by MTCNN")
            return None
        
        print(f"Face tensor shape: {face_tensor.shape}")
        
        # Handle multiple faces - use the first one
        if len(face_tensor.shape) == 4 and face_tensor.shape[0] > 1:
            print(f"Multiple faces detected ({face_tensor.shape[0]}), using the first one")
            face_tensor = face_tensor[0]
        
        # Ensure we have the right dimensions [1, 3, 160, 160]
        if len(face_tensor.shape) == 3:
            face_tensor = face_tensor.unsqueeze(0)
        
        face_tensor = face_tensor.to(device)
        print(f"Face tensor moved to device: {face_tensor.shape}")
        
        # Get embedding using FaceNet
        with torch.no_grad():
            embedding = facenet(face_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
        
        embedding_np = embedding.cpu().numpy().flatten()
        print(f"Generated embedding of size: {embedding_np.shape}")
        
        return embedding_np
        
    except Exception as e:
        print(f"Face embedding extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_faces(embedding1, embedding2, threshold=0.55):
    """Compare two face embeddings using cosine similarity with improved accuracy"""
    try:
        # Ensure embeddings are numpy arrays
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2)
        
        # Normalize embeddings (they should already be normalized from FaceNet)
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        # Ensure similarity is in valid range [-1, 1]
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # Convert to positive similarity score [0, 1]
        similarity_score = (similarity + 1) / 2
        
        # Apply confidence boost for high similarities
        if similarity_score > 0.80:
            similarity_score = min(similarity_score * 1.08, 1.0)
        elif similarity_score > 0.70:
            similarity_score = min(similarity_score * 1.05, 1.0)
        
        is_match = similarity_score > threshold
        
        return is_match, float(similarity_score)
        
    except Exception as e:
        print(f"Face comparison error: {e}")
        return False, 0.0

def load_face_encodings():
    """Load face encodings from database with student details"""
    try:
        conn = get_db_connection()
        if conn is None:
            return [], [], []
        
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                fe.student_id, 
                fe.face_encoding,
                s.name,
                s.roll_no,
                s.course,
                s.year,
                s.division
            FROM face_encodings fe
            JOIN student s ON fe.student_id = s.id
            ORDER BY s.roll_no
        """)
        results = cursor.fetchall()
        
        encodings = []
        student_ids = []
        student_details = []
        
        for result in results:
            if result['face_encoding']:
                try:
                    # Deserialize the encoding
                    encoding = pickle.loads(result['face_encoding'])
                    encodings.append(encoding)
                    student_ids.append(result['student_id'])
                    student_details.append({
                        'id': result['student_id'],
                        'name': result['name'],
                        'roll_no': result['roll_no'],
                        'course': result['course'],
                        'year': result['year'],
                        'division': result['division']
                    })
                except Exception as e:
                    print(f"Error deserializing encoding for student {result['student_id']}: {e}")
                    continue
        
        cursor.close()
        conn.close()
        
        print(f"Loaded {len(encodings)} face encodings from database")
        return encodings, student_ids, student_details
        
    except Exception as e:
        print(f"Error loading face encodings: {e}")
        return [], [], []

def save_face_encoding(student_id, encoding):
    """Save face encoding to database"""
    try:
        conn = get_db_connection()
        if conn is None:
            return False
        
        cursor = conn.cursor()
        
        # Serialize the encoding
        encoding_blob = pickle.dumps(encoding)
        
        # Check if encoding already exists
        cursor.execute("SELECT id FROM face_encodings WHERE student_id = %s", (student_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing encoding
            cursor.execute(
                "UPDATE face_encodings SET face_encoding = %s, updated_at = %s WHERE student_id = %s",
                (encoding_blob, datetime.now(), student_id)
            )
            print(f"Updated face encoding for student ID: {student_id}")
        else:
            # Insert new encoding
            cursor.execute(
                "INSERT INTO face_encodings (student_id, face_encoding, created_at) VALUES (%s, %s, %s)",
                (student_id, encoding_blob, datetime.now())
            )
            print(f"Inserted new face encoding for student ID: {student_id}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error saving face encoding: {e}")
        return False

def save_face_encoding_with_details(student_id, encoding, student_details):
    """Save face encoding to database with student details for better tracking"""
    try:
        conn = get_db_connection()
        if conn is None:
            return False
        
        cursor = conn.cursor()
        
        # Serialize the encoding
        encoding_blob = pickle.dumps(encoding)
        
        # Check if encoding already exists
        cursor.execute("SELECT id FROM face_encodings WHERE student_id = %s", (student_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing encoding
            cursor.execute(
                "UPDATE face_encodings SET face_encoding = %s, updated_at = %s WHERE student_id = %s",
                (encoding_blob, datetime.now(), student_id)
            )
            print(f"Updated face encoding for {student_details['name']} (Roll No: {student_details['roll_no']})")
        else:
            # Insert new encoding
            cursor.execute(
                "INSERT INTO face_encodings (student_id, face_encoding, created_at) VALUES (%s, %s, %s)",
                (student_id, encoding_blob, datetime.now())
            )
            print(f"Inserted new face encoding for {student_details['name']} (Roll No: {student_details['roll_no']})")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error saving face encoding for {student_details.get('name', 'Unknown')}: {e}")
        return False

@app.route('/api/auth/login', methods=['POST', 'OPTIONS'])
def login():
    """Authenticate user"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        username = data.get('username')
        password = data.get('password')
        
        print(f"Login attempt: username={username}")
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        conn = get_db_connection()
        
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # Check admin table first
        cursor.execute("SELECT * FROM admin WHERE username = %s", (username,))
        admin_user = cursor.fetchone()
        
        if admin_user:
            # For demo purposes, accept simple passwords for admin
            if password == 'admin123':
                user_data = {
                    'id': admin_user['admin_id'],
                    'username': admin_user['username'],
                    'name': admin_user['full_name'],
                    'email': admin_user['email'] or '',
                    'role': 'admin',
                    'subjects': []  # Admin has access to all subjects
                }
                
                cursor.close()
                conn.close()
                return jsonify(user_data)
        
        # Check teachers table
        cursor.execute("SELECT * FROM teachers WHERE username = %s", (username,))
        teacher_user = cursor.fetchone()
        
        if teacher_user:
            # Check if password matches the one in database
            stored_password = teacher_user.get('password', '')
            demo_passwords = ['teacher123', 'password', 'Evo01', 'Evo02', 'A_011']
            
            # If a password is set in database, only use that password
            if stored_password and stored_password.strip():
                password_valid = password == stored_password
            else:
                # If no password set in database, use demo passwords for backward compatibility
                password_valid = password in demo_passwords
            
            if password_valid:
                # Get teacher's subjects
                subjects = []
                if teacher_user['subject_id']:
                    cursor.execute("SELECT subject_name FROM subjects WHERE subject_id = %s", (teacher_user['subject_id'],))
                    subject_result = cursor.fetchone()
                    if subject_result:
                        subjects.append(subject_result['subject_name'])
                
                # Also check teacher_subjects junction table
                cursor.execute("""
                    SELECT s.subject_name 
                    FROM subjects s
                    JOIN teacher_subjects ts ON s.subject_id = ts.subject_id
                    WHERE ts.teacher_id = %s
                """, (teacher_user['teacher_id'],))
                
                junction_subjects = [row['subject_name'] for row in cursor.fetchall()]
                subjects.extend(junction_subjects)
                
                # Remove duplicates
                subjects = list(set(subjects))
                
                user_data = {
                    'id': teacher_user['teacher_id'],
                    'username': teacher_user['username'],
                    'name': teacher_user['name'],
                    'email': '',  # Teachers table doesn't have email in your schema
                    'role': 'teacher',
                    'subjects': subjects
                }
                
                cursor.close()
                conn.close()
                return jsonify(user_data)
        
        # Check students table
        cursor.execute("SELECT * FROM student WHERE roll_no = %s", (username,))
        student_user = cursor.fetchone()
        
        if student_user:
            # Student password is always '123'
            if password == '123':
                user_data = {
                    'id': student_user['id'],
                    'username': student_user['roll_no'],
                    'name': student_user['name'],
                    'roll_no': student_user['roll_no'],
                    'course': student_user['course'],
                    'year': student_user['year'],
                    'division': student_user['division'],
                    'subdivision': student_user['subdivision'],
                    'role': 'student'
                }
                
                cursor.close()
                conn.close()
                return jsonify(user_data)
        
        cursor.close()
        conn.close()
        return jsonify({'error': 'Invalid credentials'}), 401
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance', methods=['GET'])
def get_attendance_records():
    """Get attendance records with optional filters"""
    try:
        subject = request.args.get('subject')
        date_filter = request.args.get('date')
        student_id = request.args.get('student_id')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT 
                a.student_id,
                s.name as student_name,
                s.roll_no,
                s.course,
                s.year,
                s.division,
                s.subdivision,
                sub.subject_name as subject,
                a.attendance_date as date,
                a.attendance_time as time,
                a.status,
                a.marked_by,
                t.name as marked_by_teacher,
                a.recognition_confidence,
                asess.session_name,
                asess.start_time as session_start_time,
                asess.end_time as session_end_time,
                asess.is_active as session_active
            FROM attendance a
            JOIN student s ON a.student_id = s.id
            JOIN subjects sub ON a.subject_id = sub.subject_id
            LEFT JOIN teachers t ON a.marked_by = t.teacher_id
            LEFT JOIN attendance_sessions asess ON a.session_id = asess.id
            WHERE 1=1
        """
        
        params = []
        
        if subject:
            query += " AND sub.subject_name = %s"
            params.append(subject)
            
        if date_filter:
            query += " AND DATE(a.attendance_date) = %s"
            params.append(date_filter)
            
        if student_id:
            query += " AND a.student_id = %s"
            params.append(student_id)
            
        query += " ORDER BY a.attendance_date DESC, a.attendance_time DESC LIMIT 100"
        
        cursor.execute(query, params)
        records = cursor.fetchall()
        
        # Convert date and time objects to strings
        for record in records:
            if isinstance(record['date'], date):
                record['date'] = record['date'].isoformat()
            if record['time']:
                record['time'] = str(record['time'])
        
        cursor.close()
        conn.close()
        
        return jsonify({'records': records, 'count': len(records)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/mark', methods=['POST'])
def mark_attendance():
    """Mark attendance for a student (teacher-initiated)"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        subject = data.get('subject')
        status = data.get('status', 'Present')
        attendance_date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        attendance_time = data.get('time', datetime.now().strftime('%H:%M:%S'))
        marked_by = data.get('marked_by')  # Teacher ID
        recognition_confidence = data.get('confidence', 0.0)
        
        if not student_id or not subject:
            return jsonify({'error': 'Student ID and subject required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Verify student exists
        cursor.execute("SELECT id, name, roll_no FROM student WHERE id = %s", (student_id,))
        student = cursor.fetchone()
        
        if not student:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Student not found'}), 404
        
        # Get subject ID
        cursor.execute("SELECT subject_id FROM subjects WHERE subject_name = %s", (subject,))
        subject_result = cursor.fetchone()
        
        if not subject_result:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Subject not found'}), 404
        
        subject_id = subject_result['subject_id']
        
        # Check if attendance already exists for today
        cursor.execute("""
            SELECT id FROM attendance 
            WHERE student_id = %s AND subject_id = %s AND attendance_date = %s
        """, (student_id, subject_id, attendance_date))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            cursor.execute("""
                UPDATE attendance 
                SET attendance_time = %s, status = %s, marked_by = %s, recognition_confidence = %s
                WHERE id = %s
            """, (attendance_time, status, marked_by, recognition_confidence, existing['id']))
            message = f"Attendance updated for {student['name']} ({student['roll_no']})"
        else:
            # Insert new record
            cursor.execute("""
                INSERT INTO attendance (student_id, subject_id, attendance_date, attendance_time, status, marked_by, recognition_confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (student_id, subject_id, attendance_date, attendance_time, status, marked_by, recognition_confidence))
            message = f"Attendance marked for {student['name']} ({student['roll_no']})"
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': message,
            'student_name': student['name'],
            'roll_no': student['roll_no'],
            'status': status,
            'confidence': recognition_confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/mark-student', methods=['POST'])
def mark_student_attendance():
    """Mark attendance for a student with location verification"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        session_id = data.get('session_id')
        student_latitude = data.get('latitude')
        student_longitude = data.get('longitude')
        recognition_confidence = data.get('confidence', 0.0)
        
        if not all([student_id, session_id, student_latitude, student_longitude]):
            return jsonify({'error': 'Student ID, session ID, and location are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get session details
        cursor.execute("""
            SELECT s.*, sub.subject_name, sub.subject_id
            FROM attendance_sessions s
            JOIN subjects sub ON s.subject_id = sub.subject_id
            WHERE s.id = %s AND s.is_active = TRUE
        """, (session_id,))
        
        session = cursor.fetchone()
        
        if not session:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Session not found or not active'}), 404
        
        # Calculate distance between student and teacher location
        distance = calculate_distance(
            float(session['latitude']), float(session['longitude']),
            float(student_latitude), float(student_longitude)
        )
        
        # GPS accuracy tolerance: Add buffer for GPS inaccuracy
        # Most mobile GPS has 10-50m accuracy, so we add a tolerance
        gps_tolerance = 40  # meters - increased to 40m for better reliability
        effective_radius = session['radius_meters'] + gps_tolerance
        
        print(f"Distance from session location: {distance:.2f}m (radius: {session['radius_meters']}m + tolerance: {gps_tolerance}m = {effective_radius}m)")
        
        if distance > effective_radius:
            cursor.close()
            conn.close()
            return jsonify({
                'error': f'You are too far from the class location. Distance: {distance:.0f}m, Required: within {session["radius_meters"]}m (±{gps_tolerance}m GPS tolerance)',
                'distance': distance,
                'required_distance': session['radius_meters'],
                'gps_tolerance': gps_tolerance,
                'effective_radius': effective_radius
            }), 400
        
        # Get student details
        cursor.execute("SELECT id, name, roll_no FROM student WHERE id = %s", (student_id,))
        student = cursor.fetchone()
        
        if not student:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Student not found'}), 404
        
        # Check if attendance already marked for this session
        cursor.execute("""
            SELECT student_id FROM attendance 
            WHERE student_id = %s AND session_id = %s
        """, (student_id, session_id))
        
        existing = cursor.fetchone()
        
        if existing:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Attendance already marked for this session'}), 400
        
        # Mark attendance
        attendance_date = datetime.now().strftime('%Y-%m-%d')
        attendance_time = datetime.now().strftime('%H:%M:%S')
        
        cursor.execute("""
            INSERT INTO attendance 
            (student_id, subject_id, session_id, attendance_date, attendance_time, status, recognition_confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (student_id, session['subject_id'], session_id, attendance_date, attendance_time, 'Present', recognition_confidence))
        
        print(f"✅ Marked PRESENT: Student ID {student_id} ({student['roll_no']} - {student['name']}) for session {session_id}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': f'Attendance marked successfully for {student["name"]}',
            'student_name': student['name'],
            'roll_no': student['roll_no'],
            'subject': session['subject_name'],
            'session_name': session['session_name'],
            'distance': distance,
            'confidence': recognition_confidence
        })
        
    except Exception as e:
        print(f"Mark student attendance error: {e}")
        return jsonify({'error': str(e)}), 500

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters using Haversine formula"""
    import math
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    
    return c * r

@app.route('/api/students', methods=['GET'])
def get_students():
    """Get all students with face encoding status"""
    try:
        course = request.args.get('course')
        year = request.args.get('year')
        division = request.args.get('division')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT 
                s.id, 
                s.name, 
                s.roll_no, 
                s.course, 
                s.year, 
                s.division, 
                s.subdivision, 

                CASE WHEN fe.student_id IS NOT NULL THEN 1 ELSE 0 END as has_face_encoding
            FROM student s
            LEFT JOIN face_encodings fe ON s.id = fe.student_id
            WHERE 1=1
        """
        
        params = []
        
        if course:
            query += " AND s.course = %s"
            params.append(course)
            
        if year:
            query += " AND s.year = %s"
            params.append(year)
            
        if division:
            query += " AND s.division = %s"
            params.append(division)
        
        query += " ORDER BY s.course, s.year, s.division, s.name"
        
        cursor.execute(query, params)
        students = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'students': students,
            'count': len(students),
            'trained_count': sum(1 for s in students if s['has_face_encoding'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    """Get all subjects with teacher assignments"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                s.subject_id, 
                s.subject_name,
                GROUP_CONCAT(t.name SEPARATOR ', ') as teachers
            FROM subjects s
            LEFT JOIN teacher_subjects ts ON s.subject_id = ts.subject_id
            LEFT JOIN teachers t ON ts.teacher_id = t.teacher_id
            GROUP BY s.subject_id, s.subject_name
            ORDER BY s.subject_name
        """)
        subjects = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({'subjects': subjects, 'count': len(subjects)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:student_id>', methods=['GET'])
def get_student_details(student_id):
    """Get detailed student information"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get student details
        cursor.execute("""
            SELECT 
                s.*,
                CASE WHEN fe.student_id IS NOT NULL THEN 1 ELSE 0 END as has_face_encoding,
                fe.created_at as face_trained_at
            FROM student s
            LEFT JOIN face_encodings fe ON s.id = fe.student_id
            WHERE s.id = %s
        """, (student_id,))
        
        student = cursor.fetchone()
        
        if not student:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Student not found'}), 404
        
        # Get attendance statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_classes,
                SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as present_count,
                SUM(CASE WHEN status = 'Absent' THEN 1 ELSE 0 END) as absent_count,
                SUM(CASE WHEN status = 'Late' THEN 1 ELSE 0 END) as late_count
            FROM attendance
            WHERE student_id = %s
        """, (student_id,))
        
        stats = cursor.fetchone()
        
        # Get recent attendance
        cursor.execute("""
            SELECT 
                a.attendance_date,
                a.attendance_time,
                a.status,
                s.subject_name,
                a.recognition_confidence
            FROM attendance a
            JOIN subjects s ON a.subject_id = s.subject_id
            WHERE a.student_id = %s
            ORDER BY a.attendance_date DESC, a.attendance_time DESC
            LIMIT 10
        """, (student_id,))
        
        recent_attendance = cursor.fetchall()
        
        # Convert dates to strings
        for record in recent_attendance:
            if isinstance(record['attendance_date'], date):
                record['attendance_date'] = record['attendance_date'].isoformat()
            if record['attendance_time']:
                record['attendance_time'] = str(record['attendance_time'])
        
        cursor.close()
        conn.close()
        
        # Calculate attendance percentage
        total = stats['total_classes'] or 1
        attendance_percentage = (stats['present_count'] / total) * 100
        
        return jsonify({
            'student': student,
            'statistics': {
                **stats,
                'attendance_percentage': round(attendance_percentage, 2)
            },
            'recent_attendance': recent_attendance
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/face-recognition/encodings', methods=['GET'])
def get_face_encodings():
    """Get face encodings for recognition with student details"""
    try:
        encodings, student_ids, student_details = load_face_encodings()
        
        # Convert numpy arrays to lists for JSON serialization (optional, for debugging)
        # encodings_list = [encoding.tolist() if isinstance(encoding, np.ndarray) else encoding 
        #                  for encoding in encodings]
        
        return jsonify({
            'count': len(encodings),
            'students': student_details,
            'message': f'Found {len(encodings)} trained face encodings'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/face-recognition/trained-faces', methods=['GET'])
def get_trained_faces():
    """Get list of students with trained faces"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                s.id,
                s.name,
                s.roll_no,
                s.course,
                s.year,
                s.division,
                s.subdivision,
                fe.created_at as trained_at,
                fe.updated_at as last_updated
            FROM student s
            JOIN face_encodings fe ON s.id = fe.student_id
            ORDER BY s.roll_no
        """)
        
        trained_faces = cursor.fetchall()
        
        # Convert datetime objects to strings
        for face in trained_faces:
            if face['trained_at']:
                face['trained_at'] = face['trained_at'].isoformat()
            if face['last_updated']:
                face['last_updated'] = face['last_updated'].isoformat()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'trained_faces': trained_faces,
            'count': len(trained_faces)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/face-recognition/recognize', methods=['POST'])
def recognize_face():
    """Recognize face from image bytes using MTCNN and FaceNet with stored embeddings"""
    try:
        print("=" * 60)
        print("FACE RECOGNITION REQUEST RECEIVED")
        print("=" * 60)
        
        # Get image data
        image_data = request.get_data()
        
        if not image_data:
            print("❌ No image data provided")
            return jsonify({'error': 'No image data provided'}), 400
        
        print(f"✓ Received image data: {len(image_data)} bytes")
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        print(f"Processing image of shape: {img.shape}")
        
        # Extract face embedding from input image using MTCNN and FaceNet
        input_embedding = extract_face_embedding(img)
        
        if input_embedding is None:
            return jsonify({'error': 'No face found in image. Please ensure face is clearly visible.'}), 404
        
        print(f"Extracted input embedding of size: {len(input_embedding)}")
        
        # Load known face encodings from database with student details
        known_encodings, known_ids, student_details = load_face_encodings()
        
        if not known_encodings:
            return jsonify({'error': 'No trained face encodings found. Please train faces first.'}), 404
        
        print(f"Comparing against {len(known_encodings)} stored face encodings")
        
        # Find best match using cosine similarity with improved threshold
        best_match_idx = None
        best_similarity = 0.0
        second_best_similarity = 0.0
        threshold = 0.55  # Lowered threshold for better recognition (was 0.65)
        
        similarities = []
        for i, known_encoding in enumerate(known_encodings):
            is_match, similarity = compare_faces(input_embedding, known_encoding, threshold)
            similarities.append((i, similarity))
            
            print(f"Student {student_details[i]['roll_no']} ({student_details[i]['name']}): similarity = {similarity:.3f}")
            
            if similarity > best_similarity:
                second_best_similarity = best_similarity
                best_similarity = similarity
                best_match_idx = i
            elif similarity > second_best_similarity:
                second_best_similarity = similarity
        
        # Additional validation: ensure best match is significantly better than second best
        confidence_margin = 0.03  # 3% margin (reduced from 5% for better recognition)
        if best_match_idx is not None and (best_similarity - second_best_similarity) < confidence_margin and second_best_similarity > 0.5:
            print(f"Warning: Best match similarity ({best_similarity:.3f}) too close to second best ({second_best_similarity:.3f})")
            # Still proceed but with lower confidence
        
        # Check if best match exceeds threshold
        if best_match_idx is not None and best_similarity > threshold:
            matched_student = student_details[best_match_idx]
            confidence_level = "High" if best_similarity > 0.8 else "Medium" if best_similarity > 0.7 else "Low"
            print(f"Face recognized: {matched_student['name']} (Roll No: {matched_student['roll_no']}) with confidence {best_similarity:.3f} ({confidence_level})")
            
            return jsonify({
                'student_id': int(matched_student['id']),
                'confidence': float(best_similarity),
                'student_name': matched_student['name'],
                'roll_no': matched_student['roll_no'],
                'course': matched_student['course'],
                'year': matched_student['year'],
                'division': matched_student['division'],
                'threshold_used': threshold,
                'confidence_level': confidence_level,
                'margin_to_second': float(best_similarity - second_best_similarity)
            })
        
        print("=" * 60)
        print(f"❌ NO FACE MATCH FOUND")
        print(f"Best similarity: {best_similarity:.3f} (threshold: {threshold})")
        print(f"All similarities: {[f'{s:.3f}' for _, s in sorted(similarities, key=lambda x: x[1], reverse=True)[:5]]}")
        print("=" * 60)
        return jsonify({'error': 'Face not recognized. Please ensure you are registered in the system.'}), 404
        
    except Exception as e:
        print(f"Face recognition error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/face-recognition/train', methods=['POST'])
def train_face():
    """Train face recognition with new face data using MTCNN and FaceNet with duplicate detection"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        image_data = data.get('image_data')  # Base64 encoded image
        
        if not student_id or not image_data:
            return jsonify({'error': 'Student ID and image data required'}), 400
        
        # Verify student exists and get details
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, roll_no, course, year, division FROM student WHERE id = %s", (student_id,))
        student = cursor.fetchone()
        
        if not student:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Student not found'}), 404
        
        # Check if student already has face trained
        cursor.execute("SELECT student_id FROM face_encodings WHERE student_id = %s", (student_id,))
        existing = cursor.fetchone()
        
        if existing:
            print(f"Student {student['name']} (Roll No: {student['roll_no']}) already has face trained")
            # Allow re-training but log it
            print(f"Re-training face for student: {student['name']} (Roll No: {student['roll_no']})")
        else:
            print(f"Training face for student: {student['name']} (Roll No: {student['roll_no']})")
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Extract face embedding using MTCNN and FaceNet
        face_embedding = extract_face_embedding(img)
        
        if face_embedding is None:
            cursor.close()
            conn.close()
            return jsonify({'error': 'No face found in image. Please ensure face is clearly visible and well-lit.'}), 400
        
        print(f"Generated face embedding of size: {len(face_embedding)}")
        
        # ANTI-PROXY MEASURE: Check if this face already exists for another student
        print("Checking for duplicate faces (anti-proxy measure)...")
        known_encodings, known_ids, student_details = load_face_encodings()
        
        duplicate_threshold = 0.75  # High similarity threshold for duplicate detection
        for i, known_encoding in enumerate(known_encodings):
            # Skip if it's the same student (re-training)
            if known_ids[i] == student_id:
                continue
            
            is_match, similarity = compare_faces(face_embedding, known_encoding, duplicate_threshold)
            
            if is_match or similarity > duplicate_threshold:
                duplicate_student = student_details[i]
                print(f"⚠️ DUPLICATE FACE DETECTED! This face matches {duplicate_student['name']} (Roll No: {duplicate_student['roll_no']}) with {similarity:.3f} similarity")
                cursor.close()
                conn.close()
                return jsonify({
                    'error': f'This face is already registered to another student ({duplicate_student["name"]}, Roll No: {duplicate_student["roll_no"]}). Proxy attendance is not allowed.',
                    'duplicate_detected': True,
                    'duplicate_student': duplicate_student['name'],
                    'duplicate_roll_no': duplicate_student['roll_no'],
                    'similarity': float(similarity)
                }), 403  # 403 Forbidden
        
        print("✓ No duplicate faces found. Proceeding with training...")
        
        # Save face encoding to database with student details
        success = save_face_encoding_with_details(student_id, face_embedding, student)
        
        cursor.close()
        conn.close()
        
        if success:
            return jsonify({
                'message': f'Face trained successfully for {student["name"]} (Roll No: {student["roll_no"]})',
                'student_name': student['name'],
                'roll_no': student['roll_no'],
                'embedding_size': len(face_embedding),
                'is_retrain': existing is not None
            })
        else:
            return jsonify({'error': 'Failed to save face encoding'}), 500
        
    except Exception as e:
        print(f"Face training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/stats', methods=['GET'])
def get_attendance_stats():
    """Get attendance statistics"""
    try:
        subject = request.args.get('subject')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        course = request.args.get('course')
        year = request.args.get('year')
        division = request.args.get('division')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Base query for statistics
        query = """
            SELECT 
                COUNT(*) as total_records,
                SUM(CASE WHEN a.status = 'Present' THEN 1 ELSE 0 END) as present_count,
                SUM(CASE WHEN a.status = 'Absent' THEN 1 ELSE 0 END) as absent_count,
                SUM(CASE WHEN a.status = 'Late' THEN 1 ELSE 0 END) as late_count,
                COUNT(DISTINCT a.student_id) as unique_students,
                COUNT(DISTINCT a.attendance_date) as unique_dates,
                AVG(a.recognition_confidence) as avg_confidence
            FROM attendance a
            JOIN subjects sub ON a.subject_id = sub.subject_id
            JOIN student s ON a.student_id = s.id
            WHERE 1=1
        """
        
        params = []
        
        if subject:
            query += " AND sub.subject_name = %s"
            params.append(subject)
            
        if start_date:
            query += " AND a.attendance_date >= %s"
            params.append(start_date)
            
        if end_date:
            query += " AND a.attendance_date <= %s"
            params.append(end_date)
            
        if course:
            query += " AND s.course = %s"
            params.append(course)
            
        if year:
            query += " AND s.year = %s"
            params.append(year)
            
        if division:
            query += " AND s.division = %s"
            params.append(division)
        
        cursor.execute(query, params)
        stats = cursor.fetchone()
        
        # Calculate percentage
        total = stats['total_records'] or 1
        stats['attendance_percentage'] = round((stats['present_count'] / total) * 100, 2)
        stats['avg_confidence'] = round(stats['avg_confidence'] or 0, 3)
        
        cursor.close()
        conn.close()
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/teachers', methods=['GET'])
def get_teachers():
    """Get all teachers with their subjects"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                t.teacher_id,
                t.name,
                t.username,
                GROUP_CONCAT(s.subject_name SEPARATOR ', ') as subjects
            FROM teachers t
            LEFT JOIN teacher_subjects ts ON t.teacher_id = ts.teacher_id
            LEFT JOIN subjects s ON ts.subject_id = s.subject_id
            GROUP BY t.teacher_id, t.name, t.username
            ORDER BY t.name
        """)
        
        teachers = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({'teachers': teachers, 'count': len(teachers)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/divisions', methods=['GET'])
def get_divisions():
    """Get all divisions with student counts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                course,
                year,
                division,
                subdivision,
                COUNT(*) as student_count,
                SUM(CASE WHEN fe.student_id IS NOT NULL THEN 1 ELSE 0 END) as trained_count
            FROM student s
            LEFT JOIN face_encodings fe ON s.id = fe.student_id
            GROUP BY course, year, division, subdivision
            ORDER BY course, year, division, subdivision
        """)
        
        divisions = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({'divisions': divisions, 'count': len(divisions)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global models_initialized, mtcnn, facenet
    
    status = {
        'status': 'healthy' if models_initialized else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'mtcnn': mtcnn is not None,
            'facenet': facenet is not None,
            'device': str(device) if device else 'unknown'
        }
    }
    
    return jsonify(status)

@app.route('/api/test', methods=['GET'])
def test_connection():
    """Test connection endpoint"""
    return jsonify({'message': 'Connection successful', 'timestamp': datetime.now().isoformat()})

@app.route('/api/debug/tables', methods=['GET'])
def debug_tables():
    """Debug endpoint to check database tables"""
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Get structure of each table
        table_info = {}
        for table in tables:
            cursor.execute(f"DESCRIBE {table}")
            columns = [{'Field': row[0], 'Type': row[1], 'Null': row[2], 'Key': row[3]} for row in cursor.fetchall()]
            table_info[table] = columns
        
        cursor.close()
        conn.close()
        
        return jsonify({'tables': tables, 'structure': table_info})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/users', methods=['GET'])
def debug_users():
    """Debug endpoint to check existing users"""
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        # Check teachers table
        cursor.execute("SELECT teacher_id, username, name FROM teachers")
        teachers = cursor.fetchall()
        
        # Check admin table if it exists
        admin_users = []
        try:
            cursor.execute("SELECT * FROM admin")
            admin_users = cursor.fetchall()
        except:
            pass
        
        cursor.close()
        conn.close()
        
        return jsonify({'teachers': teachers, 'admin_users': admin_users})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/face-recognition/test-recognition', methods=['POST'])
def test_face_recognition():
    """Test face recognition against a specific student"""
    try:
        data = request.get_json()
        test_student_id = data.get('test_student_id')
        image_data = data.get('image_data')
        
        if not test_student_id or not image_data:
            return jsonify({'error': 'Test student ID and image data required'}), 400
        
        # Decode and process image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Extract embedding from test image
        test_embedding = extract_face_embedding(img)
        if test_embedding is None:
            return jsonify({'error': 'No face found in test image'}), 400
        
        # Load stored embeddings
        known_encodings, known_ids, student_details = load_face_encodings()
        
        # Find the specific student's embedding
        target_idx = None
        for i, student_id in enumerate(known_ids):
            if student_id == test_student_id:
                target_idx = i
                break
        
        if target_idx is None:
            return jsonify({'error': 'Test student not found in trained faces'}), 404
        
        # Compare with target student
        is_match, similarity = compare_faces(test_embedding, known_encodings[target_idx])
        target_student = student_details[target_idx]
        
        # Also compare with all other students for reference
        all_similarities = []
        for i, (encoding, student) in enumerate(zip(known_encodings, student_details)):
            _, sim = compare_faces(test_embedding, encoding)
            all_similarities.append({
                'student_id': student['id'],
                'name': student['name'],
                'roll_no': student['roll_no'],
                'similarity': float(sim),
                'is_target': student['id'] == test_student_id
            })
        
        # Sort by similarity
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return jsonify({
            'target_student': {
                'id': target_student['id'],
                'name': target_student['name'],
                'roll_no': target_student['roll_no']
            },
            'target_similarity': float(similarity),
            'is_match': is_match,
            'threshold': 0.6,
            'all_similarities': all_similarities[:10]  # Top 10 matches
        })
        
    except Exception as e:
        print(f"Test recognition error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/sessions', methods=['POST'])
def create_attendance_session():
    """Create a new attendance session with location"""
    try:
        data = request.get_json()
        teacher_id = data.get('teacher_id')
        subject_name = data.get('subject')
        session_name = data.get('session_name', f'Session {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        radius_meters = data.get('radius_meters', 20)
        
        if not all([teacher_id, subject_name, latitude, longitude]):
            return jsonify({'error': 'Teacher ID, subject, latitude, and longitude are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get subject ID
        cursor.execute("SELECT subject_id FROM subjects WHERE subject_name = %s", (subject_name,))
        subject_result = cursor.fetchone()
        
        if not subject_result:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Subject not found'}), 404
        
        subject_id = subject_result['subject_id']
        
        # End any existing active sessions for this teacher and subject
        cursor.execute("""
            UPDATE attendance_sessions 
            SET is_active = FALSE, end_time = %s 
            WHERE teacher_id = %s AND subject_id = %s AND is_active = TRUE
        """, (datetime.now(), teacher_id, subject_id))
        
        # Create new session
        cursor.execute("""
            INSERT INTO attendance_sessions 
            (teacher_id, subject_id, session_name, latitude, longitude, radius_meters, start_time, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (teacher_id, subject_id, session_name, latitude, longitude, radius_meters, datetime.now(), True))
        
        session_id = cursor.lastrowid
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': 'Attendance session created successfully',
            'session_id': session_id,
            'session_name': session_name,
            'subject': subject_name,
            'location': {'latitude': latitude, 'longitude': longitude},
            'radius_meters': radius_meters
        })
        
    except Exception as e:
        print(f"Create session error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/sessions/<int:session_id>/end', methods=['POST'])
def end_attendance_session(session_id):
    """End an attendance session and mark absent students"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get session details
        cursor.execute("""
            SELECT s.*, sub.subject_name, t.name as teacher_name
            FROM attendance_sessions s
            JOIN subjects sub ON s.subject_id = sub.subject_id
            JOIN teachers t ON s.teacher_id = t.teacher_id
            WHERE s.id = %s AND s.is_active = TRUE
        """, (session_id,))
        
        session = cursor.fetchone()
        if not session:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Session not found or already ended'}), 404
        
        print(f"Ending session: {session['session_name']} for {session['subject_name']}")
        
        # Get all students who should be in this session (all students for now, can be filtered by course/division later)
        cursor.execute("""
            SELECT id, name, roll_no, course, year, division, subdivision
            FROM student
            WHERE course IS NOT NULL
            ORDER BY roll_no
        """)
        all_students = cursor.fetchall()
        
        # Get students who already marked attendance in this session
        cursor.execute("""
            SELECT DISTINCT student_id, status
            FROM attendance
            WHERE session_id = %s
        """, (session_id,))
        attendance_records = cursor.fetchall()
        present_students = {row['student_id'] for row in attendance_records}
        
        print(f"Total students: {len(all_students)}, Already marked: {len(present_students)}")
        print(f"Present students IDs: {present_students}")
        for record in attendance_records:
            print(f"  Student ID {record['student_id']}: {record['status']}")
        
        # Mark absent students
        absent_count = 0
        current_time = datetime.now()
        
        for student in all_students:
            if student['id'] not in present_students:
                # Mark as absent
                cursor.execute("""
                    INSERT INTO attendance (student_id, subject_id, session_id, attendance_date, attendance_time, status, marked_by, recognition_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    student['id'],
                    session['subject_id'],
                    session_id,
                    current_time.date(),
                    current_time.time(),
                    'Absent',
                    session['teacher_id'],
                    0.0
                ))
                absent_count += 1
                print(f"Marked absent: {student['roll_no']} - {student['name']}")
        
        # End the session
        cursor.execute("""
            UPDATE attendance_sessions 
            SET is_active = FALSE, end_time = %s 
            WHERE id = %s
        """, (current_time, session_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': 'Attendance session ended successfully',
            'session_name': session['session_name'],
            'subject': session['subject_name'],
            'total_students': len(all_students),
            'present_students': len(present_students),
            'absent_students': absent_count,
            'auto_marked_absent': absent_count
        })
        
    except Exception as e:
        print(f"Error ending session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/sessions/active', methods=['GET'])
def get_active_sessions():
    """Get all active attendance sessions"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                s.id,
                s.session_name,
                s.latitude,
                s.longitude,
                s.radius_meters,
                s.start_time,
                sub.subject_name,
                t.name as teacher_name,
                COUNT(a.student_id) as attendance_count
            FROM attendance_sessions s
            JOIN subjects sub ON s.subject_id = sub.subject_id
            JOIN teachers t ON s.teacher_id = t.teacher_id
            LEFT JOIN attendance a ON s.id = a.session_id
            WHERE s.is_active = TRUE
            GROUP BY s.id, s.session_name, s.latitude, s.longitude, s.radius_meters, s.start_time, sub.subject_name, t.name
            ORDER BY s.start_time DESC
        """)
        
        sessions = cursor.fetchall()
        
        # Convert datetime to string
        for session in sessions:
            if session['start_time']:
                session['start_time'] = session['start_time'].isoformat()
        
        cursor.close()
        conn.close()
        
        return jsonify({'active_sessions': sessions, 'count': len(sessions)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/sessions/student/<int:student_id>', methods=['GET'])
def get_student_available_sessions(student_id):
    """Get available sessions for a student based on their course/division"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get student details
        cursor.execute("SELECT course, year, division FROM student WHERE id = %s", (student_id,))
        student = cursor.fetchone()
        
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        # Get active sessions (for now, show all active sessions)
        # In future, you can filter by course/division if needed
        cursor.execute("""
            SELECT 
                s.id,
                s.session_name,
                s.latitude,
                s.longitude,
                s.radius_meters,
                s.start_time,
                sub.subject_name,
                t.name as teacher_name,
                CASE WHEN a.student_id IS NOT NULL THEN 1 ELSE 0 END as already_marked
            FROM attendance_sessions s
            JOIN subjects sub ON s.subject_id = sub.subject_id
            JOIN teachers t ON s.teacher_id = t.teacher_id
            LEFT JOIN attendance a ON s.id = a.session_id AND a.student_id = %s
            WHERE s.is_active = TRUE
            ORDER BY s.start_time DESC
        """, (student_id,))
        
        sessions = cursor.fetchall()
        
        # Convert datetime to string
        for session in sessions:
            if session['start_time']:
                session['start_time'] = session['start_time'].isoformat()
        
        cursor.close()
        conn.close()
        
        return jsonify({'available_sessions': sessions, 'count': len(sessions)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/create-test-users', methods=['POST'])
def create_test_users():
    """Create test users for login"""
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # Insert test admin user
        try:
            cursor.execute("""
                INSERT INTO teachers (name, username, password, subject_id, division_id) 
                VALUES (%s, %s, %s, %s, %s)
            """, ('Admin User', 'admin', 'admin123', None, None))
        except mysql.connector.IntegrityError:
            # User already exists
            pass
        
        # Insert test teacher user
        try:
            cursor.execute("""
                INSERT INTO teachers (name, username, password, subject_id, division_id) 
                VALUES (%s, %s, %s, %s, %s)
            """, ('Teacher User', 'teacher1', 'teacher123', 1, 1))
        except mysql.connector.IntegrityError:
            # User already exists
            pass
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'message': 'Test users created successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/add-missing-students', methods=['POST'])
def add_missing_students():
    """Add missing students to the database"""
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # Add student 101 if not exists
        try:
            cursor.execute("""
                INSERT INTO student (name, roll_no, course, year, division, subdivision) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """, ('Student 101', '101', 'Computer Science', 3, 'A', 'A1'))
            print("Added student 101")
        except mysql.connector.IntegrityError:
            print("Student 101 already exists")
        
        # Add a few more students for completeness (115-120)
        additional_students = [
            ('Student 115', '115', 'Computer Science', 2, 'B', 'B1'),
            ('Student 116', '116', 'Information Technology', 3, 'A', 'A2'),
            ('Student 117', '117', 'Electronics Engineering', 2, 'A', 'A1'),
            ('Student 118', '118', 'Mechanical Engineering', 3, 'B', 'B2'),
            ('Student 119', '119', 'Civil Engineering', 2, 'A', 'A1'),
            ('Student 120', '120', 'Computer Science', 4, 'A', 'A1'),
        ]
        
        added_count = 0
        for student_data in additional_students:
            try:
                cursor.execute("""
                    INSERT INTO student (name, roll_no, course, year, division, subdivision) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, student_data)
                added_count += 1
                print(f"Added student {student_data[1]}")
            except mysql.connector.IntegrityError:
                print(f"Student {student_data[1]} already exists")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': f'Missing students added successfully. Added {added_count + 1} new students.',
            'added_students': ['101'] + [s[1] for s in additional_students]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/records/table', methods=['GET'])
def get_attendance_records_table():
    """Get attendance records in table format grouped by session/subject/teacher"""
    try:
        subject = request.args.get('subject')
        date_filter = request.args.get('date')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get attendance sessions with student counts
        query = """
            SELECT 
                asess.id as session_id,
                asess.session_name,
                asess.start_time,
                asess.end_time,
                asess.is_active,
                sub.subject_name,
                t.name as teacher_name,
                DATE(asess.start_time) as session_date,
                COUNT(DISTINCT a.student_id) as total_students,
                COUNT(DISTINCT CASE WHEN a.status = 'Present' THEN a.student_id END) as present_count,
                COUNT(DISTINCT CASE WHEN a.status = 'Absent' THEN a.student_id END) as absent_count,
                COUNT(DISTINCT CASE WHEN a.status = 'Late' THEN a.student_id END) as late_count
            FROM attendance_sessions asess
            JOIN subjects sub ON asess.subject_id = sub.subject_id
            JOIN teachers t ON asess.teacher_id = t.teacher_id
            LEFT JOIN attendance a ON asess.id = a.session_id
            WHERE 1=1
        """
        
        params = []
        
        if subject:
            query += " AND sub.subject_name = %s"
            params.append(subject)
            
        if date_filter:
            query += " AND DATE(asess.start_time) = %s"
            params.append(date_filter)
        
        query += """
            GROUP BY asess.id, asess.session_name, asess.start_time, asess.end_time, 
                     asess.is_active, sub.subject_name, t.name
            ORDER BY asess.start_time DESC
        """
        
        cursor.execute(query, params)
        sessions = cursor.fetchall()
        
        # For each session, get detailed student attendance
        for session in sessions:
            cursor.execute("""
                SELECT 
                    s.id as student_id,
                    s.name as student_name,
                    s.roll_no,
                    s.course,
                    s.year,
                    s.division,
                    s.subdivision,
                    COALESCE(a.status, 'Absent') as status,
                    a.attendance_time,
                    a.recognition_confidence
                FROM student s
                LEFT JOIN attendance a ON s.id = a.student_id AND a.session_id = %s
                WHERE s.course IS NOT NULL
                ORDER BY s.roll_no
            """, (session['session_id'],))
            
            session['students'] = cursor.fetchall()
            
            # Convert datetime objects to strings
            if session['start_time']:
                session['start_time'] = session['start_time'].isoformat()
            if session['end_time']:
                session['end_time'] = session['end_time'].isoformat()
            if session['session_date']:
                session['session_date'] = session['session_date'].isoformat()
                
            for student in session['students']:
                if student['attendance_time']:
                    student['attendance_time'] = str(student['attendance_time'])
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'sessions': sessions,
            'total_sessions': len(sessions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/clear-test-attendance', methods=['POST'])
def clear_test_attendance():
    """Clear old test attendance records and keep only session-based records"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete attendance records that don't have a session_id (old test data)
        cursor.execute("DELETE FROM attendance WHERE session_id IS NULL")
        deleted_count = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': f'Cleared {deleted_count} old test attendance records',
            'deleted_records': deleted_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/teachers', methods=['POST'])
def add_teacher():
    """Add a new teacher"""
    try:
        data = request.get_json()
        name = data.get('name')
        username = data.get('username')
        password = data.get('password')
        subject_name = data.get('subject_name')
        
        if not all([name, username, password]):
            return jsonify({'error': 'Name, username, and password are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if username already exists
        cursor.execute("SELECT teacher_id FROM teachers WHERE username = %s", (username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'error': 'Username already exists'}), 400
        
        # Get subject ID if provided
        subject_id = None
        if subject_name:
            cursor.execute("SELECT subject_id FROM subjects WHERE subject_name = %s", (subject_name,))
            subject_result = cursor.fetchone()
            if subject_result:
                subject_id = subject_result['subject_id']
        
        # Insert teacher
        cursor.execute("""
            INSERT INTO teachers (name, username, password, subject_id)
            VALUES (%s, %s, %s, %s)
        """, (name, username, password, subject_id))
        
        teacher_id = cursor.lastrowid
        
        # Link to subject if provided
        if subject_id:
            cursor.execute("""
                INSERT INTO teacher_subjects (teacher_id, subject_id)
                VALUES (%s, %s)
            """, (teacher_id, subject_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': 'Teacher added successfully',
            'teacher_id': teacher_id,
            'name': name,
            'username': username
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/teachers/<int:teacher_id>', methods=['DELETE'])
def delete_teacher(teacher_id):
    """Delete a teacher"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete teacher (cascade will handle related records)
        cursor.execute("DELETE FROM teachers WHERE teacher_id = %s", (teacher_id,))
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Teacher not found'}), 404
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'message': 'Teacher deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/students', methods=['POST'])
def add_student():
    """Add a new student"""
    try:
        data = request.get_json()
        name = data.get('name')
        roll_no = data.get('roll_no')
        course = data.get('course')
        year = data.get('year')
        division = data.get('division')
        subdivision = data.get('subdivision')
        
        if not all([name, roll_no, course, year]):
            return jsonify({'error': 'Name, roll number, course, and year are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if roll number already exists
        cursor.execute("SELECT id FROM student WHERE roll_no = %s", (roll_no,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'error': 'Roll number already exists'}), 400
        
        # Insert student
        cursor.execute("""
            INSERT INTO student (name, roll_no, course, year, division, subdivision)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (name, roll_no, course, year, division, subdivision))
        
        student_id = cursor.lastrowid
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': 'Student added successfully',
            'student_id': student_id,
            'name': name,
            'roll_no': roll_no
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/students/<int:student_id>', methods=['DELETE'])
def delete_student(student_id):
    """Delete a student"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Delete student (cascade will handle related records)
        cursor.execute("DELETE FROM student WHERE id = %s", (student_id,))
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Student not found'}), 404
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'message': 'Student deleted successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/teachers/<int:teacher_id>/reset-password', methods=['POST'])
def reset_teacher_password(teacher_id):
    """Reset teacher password"""
    try:
        data = request.get_json()
        new_password = data.get('new_password')
        
        if not new_password:
            return jsonify({'error': 'New password is required'}), 400
        
        print(f"Resetting password for teacher ID: {teacher_id} to: {new_password}")
        
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
            
        cursor = conn.cursor(dictionary=True)
        
        # First, check if teacher exists
        cursor.execute("SELECT teacher_id, name, username, password FROM teachers WHERE teacher_id = %s", (teacher_id,))
        teacher = cursor.fetchone()
        
        if not teacher:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Teacher not found'}), 404
        
        print(f"Found teacher: {teacher['name']} ({teacher['username']})")
        print(f"Current password: {teacher['password']}")
        
        # Update teacher password
        cursor.execute("""
            UPDATE teachers 
            SET password = %s 
            WHERE teacher_id = %s
        """, (new_password, teacher_id))
        
        rows_affected = cursor.rowcount
        print(f"Rows affected by update: {rows_affected}")
        
        if rows_affected == 0:
            cursor.close()
            conn.close()
            return jsonify({'error': 'No rows updated - teacher might not exist'}), 404
        
        conn.commit()
        print("Database commit successful")
        
        # Verify the update
        cursor.execute("SELECT password FROM teachers WHERE teacher_id = %s", (teacher_id,))
        updated_teacher = cursor.fetchone()
        print(f"Password after update: {updated_teacher['password']}")
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'message': 'Teacher password reset successfully',
            'teacher_name': teacher['name'],
            'rows_affected': rows_affected,
            'new_password_set': updated_teacher['password'] == new_password
        })
        
    except Exception as e:
        print(f"Password reset error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/students/<int:student_id>/reset-password', methods=['POST'])
def reset_student_password(student_id):
    """Reset student password (if students have passwords in future)"""
    try:
        data = request.get_json()
        new_password = data.get('new_password')
        
        if not new_password:
            return jsonify({'error': 'New password is required'}), 400
        
        # For now, students don't have passwords in the current schema
        # This endpoint is prepared for future use
        return jsonify({'message': 'Student password reset functionality not implemented yet'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/search', methods=['GET'])
def search_students():
    """Search students by name or roll number"""
    try:
        query = request.args.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Search by name or roll number
        search_pattern = f"%{query}%"
        cursor.execute("""
            SELECT 
                s.id,
                s.name,
                s.roll_no,
                s.course,
                s.year,
                s.division,
                s.subdivision,
                CASE WHEN fe.student_id IS NOT NULL THEN 1 ELSE 0 END as has_face_encoding
            FROM student s
            LEFT JOIN face_encodings fe ON s.id = fe.student_id
            WHERE s.name LIKE %s OR s.roll_no LIKE %s
            ORDER BY s.roll_no
            LIMIT 50
        """, (search_pattern, search_pattern))
        
        students = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'students': students,
            'count': len(students)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:student_id>/profile', methods=['GET'])
def get_student_profile(student_id):
    """Get detailed student profile with subjects and attendance records"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get student basic info
        cursor.execute("""
            SELECT 
                s.id,
                s.name,
                s.roll_no,
                s.course,
                s.year,
                s.division,
                s.subdivision,
                CASE WHEN fe.student_id IS NOT NULL THEN 1 ELSE 0 END as has_face_encoding,
                fe.created_at as face_encoding_date
            FROM student s
            LEFT JOIN face_encodings fe ON s.id = fe.student_id
            WHERE s.id = %s
        """, (student_id,))
        
        student = cursor.fetchone()
        
        if not student:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Student not found'}), 404
        
        # Convert datetime to string
        if student['face_encoding_date']:
            student['face_encoding_date'] = student['face_encoding_date'].isoformat()
        
        # Get subjects enrolled (based on attendance records)
        cursor.execute("""
            SELECT DISTINCT
                sub.subject_id,
                sub.subject_name,
                sub.subject_code
            FROM attendance a
            JOIN subjects sub ON a.subject_id = sub.subject_id
            WHERE a.student_id = %s
            ORDER BY sub.subject_name
        """, (student_id,))
        
        subjects = cursor.fetchall()
        
        # Get attendance statistics per subject
        attendance_stats = []
        for subject in subjects:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_classes,
                    SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as present_count,
                    SUM(CASE WHEN status = 'Absent' THEN 1 ELSE 0 END) as absent_count,
                    ROUND(
                        (SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 
                        2
                    ) as attendance_percentage
                FROM attendance
                WHERE student_id = %s AND subject_id = %s
            """, (student_id, subject['subject_id']))
            
            stats = cursor.fetchone()
            
            attendance_stats.append({
                'subject_id': subject['subject_id'],
                'subject_name': subject['subject_name'],
                'subject_code': subject['subject_code'],
                'total_classes': stats['total_classes'] or 0,
                'present_count': stats['present_count'] or 0,
                'absent_count': stats['absent_count'] or 0,
                'attendance_percentage': float(stats['attendance_percentage'] or 0)
            })
        
        # Get recent attendance records (last 20)
        cursor.execute("""
            SELECT 
                a.attendance_date as date,
                a.attendance_time as time,
                a.status,
                sub.subject_name as subject,
                sub.subject_code,
                t.name as marked_by_teacher,
                a.recognition_confidence,
                asess.session_name
            FROM attendance a
            JOIN subjects sub ON a.subject_id = sub.subject_id
            LEFT JOIN teachers t ON a.marked_by = t.teacher_id
            LEFT JOIN attendance_sessions asess ON a.session_id = asess.id
            WHERE a.student_id = %s
            ORDER BY a.attendance_date DESC, a.attendance_time DESC
            LIMIT 20
        """, (student_id,))
        
        attendance_records = cursor.fetchall()
        
        # Convert date and time to strings
        for record in attendance_records:
            if isinstance(record['date'], date):
                record['date'] = record['date'].isoformat()
            if record['time']:
                record['time'] = str(record['time'])
        
        # Calculate overall attendance
        cursor.execute("""
            SELECT 
                COUNT(*) as total_classes,
                SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) as present_count,
                ROUND(
                    (SUM(CASE WHEN status = 'Present' THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 
                    2
                ) as overall_percentage
            FROM attendance
            WHERE student_id = %s
        """, (student_id,))
        
        overall_stats = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'student': student,
            'subjects': attendance_stats,
            'recent_attendance': attendance_records,
            'overall_stats': {
                'total_classes': overall_stats['total_classes'] or 0,
                'present_count': overall_stats['present_count'] or 0,
                'overall_percentage': float(overall_stats['overall_percentage'] or 0)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:student_id>/update-profile', methods=['PUT'])
def update_student_profile():
    """Update student profile information"""
    try:
        data = request.get_json()
        student_id = data.get('student_id')
        phone = data.get('phone')
        email = data.get('email')
        profile_picture = data.get('profile_picture')  # Base64 encoded image
        
        if not student_id:
            return jsonify({'error': 'Student ID is required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build update query dynamically
        update_fields = []
        params = []
        
        if phone is not None:
            update_fields.append("phone = %s")
            params.append(phone)
        
        if email is not None:
            update_fields.append("email = %s")
            params.append(email)
        
        if profile_picture is not None:
            update_fields.append("profile_picture = %s")
            params.append(profile_picture)
        
        if not update_fields:
            cursor.close()
            conn.close()
            return jsonify({'error': 'No fields to update'}), 400
        
        params.append(student_id)
        query = f"UPDATE student SET {', '.join(update_fields)} WHERE id = %s"
        
        cursor.execute(query, params)
        
        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Student not found'}), 404
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'message': 'Profile updated successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:student_id>/change-password', methods=['POST'])
def change_student_password(student_id):
    """Change student password"""
    try:
        data = request.get_json()
        current_password = data.get('current_password')
        new_password = data.get('new_password')
        
        if not all([current_password, new_password]):
            return jsonify({'error': 'Current and new password are required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get student
        cursor.execute("SELECT id, name, password FROM student WHERE id = %s", (student_id,))
        student = cursor.fetchone()
        
        if not student:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Student not found'}), 404
        
        # Verify current password (default is '123')
        stored_password = student.get('password', '123')
        if current_password != stored_password:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Current password is incorrect'}), 401
        
        # Update password
        cursor.execute(
            "UPDATE student SET password = %s WHERE id = %s",
            (new_password, student_id)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'message': 'Password changed successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/students/<int:student_id>/profile-info', methods=['GET'])
def get_student_profile_info(student_id):
    """Get student profile information for editing"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT 
                id, name, roll_no, course, year, division, subdivision,
                phone, email, profile_picture
            FROM student
            WHERE id = %s
        """, (student_id,))
        
        student = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not student:
            return jsonify({'error': 'Student not found'}), 404
        
        return jsonify({'student': student})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/init-database', methods=['POST'])
def init_database():
    """Initialize database tables for face recognition"""
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 500
        
        cursor = conn.cursor()
        
        # Create face_encodings table if it doesn't exist
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
        
        # Add recognition_confidence column to attendance table if it doesn't exist
        try:
            cursor.execute("""
                ALTER TABLE attendance 
                ADD COLUMN recognition_confidence FLOAT DEFAULT 0.0
            """)
        except mysql.connector.Error:
            # Column already exists
            pass
        
        # Add marked_by column to attendance table if it doesn't exist
        try:
            cursor.execute("""
                ALTER TABLE attendance 
                ADD COLUMN marked_by INT,
                ADD FOREIGN KEY (marked_by) REFERENCES teachers(teacher_id)
            """)
        except mysql.connector.Error:
            # Column already exists
            pass
        
        # Create attendance_sessions table for location-based sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                teacher_id INT NOT NULL,
                subject_id INT NOT NULL,
                session_name VARCHAR(255) NOT NULL,
                latitude DECIMAL(10, 8) NOT NULL,
                longitude DECIMAL(11, 8) NOT NULL,
                radius_meters INT DEFAULT 200,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id),
                FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
            )
        """)
        
        # Add session_id to attendance table
        try:
            cursor.execute("""
                ALTER TABLE attendance 
                ADD COLUMN session_id INT,
                ADD FOREIGN KEY (session_id) REFERENCES attendance_sessions(id)
            """)
        except mysql.connector.Error:
            # Column already exists
            pass
        
        # Add phone, email, password, and profile_picture columns to student table
        try:
            cursor.execute("""
                ALTER TABLE student 
                ADD COLUMN phone VARCHAR(20),
                ADD COLUMN email VARCHAR(100),
                ADD COLUMN password VARCHAR(255) DEFAULT '123',
                ADD COLUMN profile_picture LONGTEXT
            """)
        except mysql.connector.Error:
            # Columns already exist
            pass
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'message': 'Database initialized successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def setup_adb_port_forwarding():
    """Setup ADB port forwarding for Android device"""
    import subprocess
    import platform
    import time
    
    # Add ADB to PATH automatically (Windows)
    adb_path = r"C:\Users\sahil\AppData\Local\Android\sdk\platform-tools"
    if os.path.exists(adb_path):
        os.environ['PATH'] = f"{os.environ['PATH']};{adb_path}"
    
    try:
        # Check if ADB is available and get devices
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            # Check if any device is connected
            lines = result.stdout.strip().split('\n')
            devices = [line for line in lines[1:] if line.strip() and '\tdevice' in line]
            
            if devices:
                device_info = devices[0].split('\t')[0]
                print("\n" + "="*60)
                print(f"📱 Android device detected: {device_info}")
                print("="*60)
                
                # Remove any existing reverse forwarding first
                try:
                    subprocess.run(
                        ['adb', 'reverse', '--remove', 'tcp:5000'],
                        capture_output=True,
                        timeout=3
                    )
                except:
                    pass
                
                # Setup reverse port forwarding (device -> computer)
                forward_result = subprocess.run(
                    ['adb', 'reverse', 'tcp:5000', 'tcp:5000'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if forward_result.returncode == 0:
                    print("✅ ADB reverse port forwarding setup successful!")
                    print("   Device can now access server at: http://localhost:5000")
                    print("="*60 + "\n")
                    return True
                else:
                    error_msg = forward_result.stderr.strip() if forward_result.stderr else "Unknown error"
                    print("⚠️  ADB reverse port forwarding failed:")
                    print(f"   {error_msg}")
                    print("\n💡 Troubleshooting:")
                    print("   1. Make sure USB debugging is enabled on your device")
                    print("   2. Accept the 'Allow USB debugging' prompt on your device")
                    print("   3. Try running: adb reverse tcp:5000 tcp:5000")
                    print("="*60 + "\n")
            else:
                print("\n" + "="*60)
                print("ℹ️  No Android devices connected via ADB")
                print("\n💡 To connect your device:")
                print("   1. Enable USB debugging on your Android device")
                print("   2. Connect via USB cable")
                print("   3. Accept the 'Allow USB debugging' prompt")
                print("   4. Run: adb devices")
                print("\n   Server will still run on http://0.0.0.0:5000")
                print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("ℹ️  ADB command failed")
            print("   Server will still run on http://0.0.0.0:5000")
            print("="*60 + "\n")
    except FileNotFoundError:
        print("\n" + "="*60)
        print("⚠️  ADB not found in PATH")
        print("\n💡 To install ADB:")
        print("   Windows: Download Android SDK Platform Tools")
        print("   https://developer.android.com/studio/releases/platform-tools")
        print("   Add to PATH: C:\\path\\to\\platform-tools")
        print("\n   Server will still run on http://0.0.0.0:5000")
        print("="*60 + "\n")
    except subprocess.TimeoutExpired:
        print("\n" + "="*60)
        print("⚠️  ADB command timed out")
        print("   Try restarting ADB: adb kill-server && adb start-server")
        print("   Server will still run on http://0.0.0.0:5000")
        print("="*60 + "\n")
    except Exception as e:
        print("\n" + "="*60)
        print(f"⚠️  Error setting up ADB port forwarding: {e}")
        print("   Server will still run on http://0.0.0.0:5000")
        print("="*60 + "\n")
    
    return False

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Flask Backend Server")
    print("="*60)
    
    # Setup ADB port forwarding
    adb_success = setup_adb_port_forwarding()
    
    print("🌐 Server URLs:")
    print("   - Local:    http://localhost:5000")
    print("   - Network:  http://0.0.0.0:5000")
    if adb_success:
        print("   - Android:  http://localhost:5000 ✅")
    else:
        print("   - Android:  Use your computer's IP address")
    print("="*60 + "\n")
    
    # Allow connections from any IP address
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)