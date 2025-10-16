-- SQL Script to Update Existing Sessions with New Radius
-- Run this in your MySQL database to update all active sessions

-- Update all active sessions to use 100m radius instead of 20m
UPDATE attendance_sessions 
SET radius_meters = 100 
WHERE is_active = TRUE 
  AND radius_meters < 100;

-- Verify the update
SELECT 
    id,
    session_name,
    subject_id,
    teacher_id,
    radius_meters,
    is_active,
    start_time
FROM attendance_sessions 
WHERE is_active = TRUE;

-- Optional: Update ALL sessions (including inactive ones)
-- Uncomment the following if you want to update all sessions
-- UPDATE attendance_sessions SET radius_meters = 100 WHERE radius_meters < 100;
