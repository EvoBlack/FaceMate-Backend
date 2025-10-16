# FaceMate Backend API

Flask-based backend API for FaceMate attendance system with face recognition capabilities.

## Features

- 🔐 User authentication (Admin, Teacher, Student)
- 👤 Face recognition using MTCNN and FaceNet
- 📍 Location-based attendance with GPS tolerance
- 🎓 Session management for teachers
- 📊 Attendance tracking and reporting
- 🚫 Anti-proxy security (duplicate face detection)

## Requirements

- Python 3.8+
- MySQL 5.7+
- CUDA-capable GPU (optional, for faster face recognition)

## Installation

### 1. Clone and Navigate
```bash
cd flutter_backend_api
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy example env file
copy .env.example .env

# Edit .env with your database credentials
notepad .env
```

### 5. Initialize Database
```bash
python init_database.py
```

## Running the Server

### Development
```bash
python app.py
```

### Production (with Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login

### Face Recognition
- `POST /api/face-recognition/train` - Train face for student
- `POST /api/face-recognition/recognize` - Recognize face
- `GET /api/face-recognition/trained-faces` - Get trained faces list

### Attendance
- `POST /api/attendance/mark` - Mark attendance (teacher)
- `POST /api/attendance/mark-student` - Mark attendance (student with location)
- `GET /api/attendance/records` - Get attendance records

### Sessions
- `POST /api/attendance/sessions` - Create attendance session
- `GET /api/attendance/sessions/active` - Get active sessions
- `GET /api/attendance/sessions/student/{id}` - Get available sessions for student
- `POST /api/attendance/sessions/{id}/end` - End session

### Students
- `GET /api/students` - Get all students
- `GET /api/students/{id}` - Get student details
- `GET /api/students/{id}/profile` - Get student profile with attendance

## Configuration

### GPS Tolerance
Default: 40m (can be adjusted in `app.py`)
```python
gps_tolerance = 40  # meters
```

### Face Recognition Threshold
Default: 0.55 (can be adjusted in `app.py`)
```python
threshold = 0.55  # similarity threshold
```

### Duplicate Detection Threshold
Default: 0.75 (can be adjusted in `app.py`)
```python
duplicate_threshold = 0.75  # anti-proxy threshold
```

## Database Schema

### Main Tables
- `admin` - Admin users
- `teachers` - Teacher accounts
- `student` - Student accounts
- `subjects` - Course subjects
- `face_encodings` - Face embeddings
- `attendance` - Attendance records
- `attendance_sessions` - Active sessions

## Security Features

### 1. Anti-Proxy Detection
- Checks for duplicate faces during training
- Prevents students from using others' faces
- 75% similarity threshold for duplicates

### 2. Location Verification
- GPS-based attendance marking
- 20m classroom radius + 40m GPS tolerance
- Prevents remote attendance marking

### 3. Session-Based Attendance
- Time-limited sessions
- Location-locked sessions
- Automatic absent marking on session end

## Troubleshooting

### Face Recognition Not Working
1. Check if models are loaded: Look for "All face recognition models initialized successfully"
2. Verify face encodings exist: Check `face_encodings` table
3. Check similarity scores in logs
4. Adjust threshold if needed

### Database Connection Issues
1. Verify MySQL is running
2. Check credentials in `.env`
3. Ensure database exists: `CREATE DATABASE face_recognizer;`
4. Run `init_database.py` to create tables

### GPS Accuracy Issues
1. Increase GPS tolerance if needed
2. Check device location settings
3. Ensure high accuracy mode is enabled
4. Test outdoors for better GPS signal

## Deployment

### Using Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app
```

### Using Docker
```bash
docker build -t facemate-backend .
docker run -p 5000:5000 facemate-backend
```

### Environment Variables for Production
```bash
FLASK_ENV=production
FLASK_DEBUG=0
DB_HOST=your-production-db-host
DB_USER=your-db-user
DB_PASSWORD=your-secure-password
SECRET_KEY=your-very-secure-secret-key
```

## Performance Optimization

### 1. Use GPU for Face Recognition
- Install CUDA and cuDNN
- PyTorch will automatically use GPU if available

### 2. Database Indexing
- Indexes are created automatically by `init_database.py`
- Monitor slow queries and add indexes as needed

### 3. Caching
- Face encodings are loaded once and cached
- Consider Redis for session management in production

## Monitoring

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Logs
- Check console output for detailed logs
- Face recognition attempts are logged with similarity scores
- Duplicate detection attempts are logged

## Support

For issues and questions:
1. Check logs for error messages
2. Verify all dependencies are installed
3. Ensure database is properly configured
4. Check face recognition model initialization

## License

Proprietary - FaceMate Attendance System
