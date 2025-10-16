# Push Backend to GitHub - Guide

## Repository
**URL:** https://github.com/EvoBlack/FaceMate-Backend.git

## Files Included in Backend

### Core Application
- `app.py` - Main Flask application with all API endpoints
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variables template
- `.gitignore` - Git ignore rules

### Database
- `init_database.py` - Database initialization script
- `UPDATE_EXISTING_SESSIONS.sql` - SQL script for updating sessions
- `update_session_radius.py` - Python script to update session radius

### Deployment
- `Dockerfile` - Docker container configuration
- `docker-compose.yml` - Docker Compose setup
- `gunicorn_config.py` - Gunicorn production server config

### Documentation
- `README.md` - Complete backend documentation
- `DEPLOYMENT.md` - Deployment instructions
- `GITHUB_PUSH_GUIDE.md` - This file

### Scripts
- `start_server.bat` - Windows server startup script
- `PUSH_TO_GITHUB.bat` - GitHub push automation script

## Quick Push (Automated)

### Option 1: Using the Batch Script
```bash
cd flutter_backend_api
PUSH_TO_GITHUB.bat
```

This will:
1. Initialize Git repository
2. Add remote origin
3. Add all files
4. Create commit
5. Push to GitHub

## Manual Push (Step by Step)

### Step 1: Navigate to Backend Folder
```bash
cd flutter_backend_api
```

### Step 2: Initialize Git
```bash
git init
```

### Step 3: Add Remote Repository
```bash
git remote add origin https://github.com/EvoBlack/FaceMate-Backend.git
```

### Step 4: Add Files
```bash
git add .
```

### Step 5: Create Commit
```bash
git commit -m "Initial commit: FaceMate Backend API with face recognition"
```

### Step 6: Push to GitHub
```bash
git branch -M main
git push -u origin main --force
```

## Important Notes

### Before Pushing

1. **Remove Sensitive Data:**
   - `.env` file is already in `.gitignore`
   - `face_encodings.pkl` is already in `.gitignore`
   - Verify no passwords or API keys in code

2. **Update .env.example:**
   - Ensure it has all required variables
   - No actual credentials in example file

3. **Check .gitignore:**
   - All sensitive files are excluded
   - Virtual environment folders excluded
   - Cache files excluded

### After Pushing

1. **Clone and Test:**
   ```bash
   git clone https://github.com/EvoBlack/FaceMate-Backend.git
   cd FaceMate-Backend
   ```

2. **Setup Environment:**
   ```bash
   cp .env.example .env
   # Edit .env with actual credentials
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Database:**
   ```bash
   python init_database.py
   ```

5. **Run Server:**
   ```bash
   python app.py
   ```

## Repository Structure

```
FaceMate-Backend/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
├── README.md                       # Documentation
├── DEPLOYMENT.md                   # Deployment guide
├── Dockerfile                      # Docker config
├── docker-compose.yml              # Docker Compose
├── gunicorn_config.py              # Production server config
├── init_database.py                # Database setup
├── update_session_radius.py        # Utility script
├── UPDATE_EXISTING_SESSIONS.sql    # SQL utility
└── start_server.bat                # Windows startup
```

## Troubleshooting

### Authentication Failed
```bash
# Use GitHub Personal Access Token
git remote set-url origin https://YOUR_TOKEN@github.com/EvoBlack/FaceMate-Backend.git
```

### Repository Already Exists
```bash
# Force push (careful - overwrites remote)
git push -u origin main --force
```

### Large Files Error
```bash
# Check file sizes
git ls-files -s | sort -k4 -n -r | head -10

# Remove large files from git
git rm --cached large_file.pkl
git commit -m "Remove large file"
```

### Permission Denied
- Ensure you have write access to the repository
- Check if repository is public or private
- Verify GitHub credentials

## Security Checklist

Before pushing, verify:
- [ ] No `.env` file in repository
- [ ] No database passwords in code
- [ ] No API keys or secrets
- [ ] No `face_encodings.pkl` (contains biometric data)
- [ ] `.gitignore` is properly configured
- [ ] `.env.example` has no real credentials

## Next Steps After Push

1. **Add Repository Description:**
   - Go to GitHub repository settings
   - Add description: "Flask backend API for FaceMate attendance system with face recognition"

2. **Add Topics:**
   - flask
   - face-recognition
   - attendance-system
   - python
   - mysql
   - pytorch

3. **Create README Badges:**
   - Python version
   - License
   - Build status

4. **Setup GitHub Actions (Optional):**
   - Automated testing
   - Docker image building
   - Deployment automation

## Support

If you encounter issues:
1. Check GitHub repository exists
2. Verify you have write access
3. Ensure Git is installed
4. Check internet connection
5. Review error messages carefully

## Success!

Once pushed, your backend will be available at:
https://github.com/EvoBlack/FaceMate-Backend.git

You can now:
- Clone it on any machine
- Deploy to cloud services
- Share with team members
- Setup CI/CD pipelines
