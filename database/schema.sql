-- =============================================================================
-- Industrial Safety Monitoring System — Database Schema
-- =============================================================================

-- Workers table: tracks each detected and tracked worker
CREATE TABLE IF NOT EXISTS workers (
    worker_id INTEGER PRIMARY KEY,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_violations INTEGER DEFAULT 0,
    total_frames_tracked INTEGER DEFAULT 0
);

-- Violations log: records each PPE violation event
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id INTEGER,
    violation_type TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    frame_number INTEGER,
    confidence REAL DEFAULT 0.0,
    snapshot_path TEXT,
    resolved BOOLEAN DEFAULT 0,
    session_id INTEGER,
    FOREIGN KEY (worker_id) REFERENCES workers(worker_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Sessions: one row per monitoring run
CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    video_source TEXT,
    total_frames INTEGER DEFAULT 0,
    total_detections INTEGER DEFAULT 0,
    total_violations INTEGER DEFAULT 0
);

-- Compliance snapshots: periodic compliance rate recordings
CREATE TABLE IF NOT EXISTS compliance_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_workers INTEGER DEFAULT 0,
    compliant_workers INTEGER DEFAULT 0,
    compliance_rate REAL DEFAULT 0.0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_violations_worker ON violations(worker_id);
CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp);
CREATE INDEX IF NOT EXISTS idx_violations_session ON violations(session_id);
CREATE INDEX IF NOT EXISTS idx_violations_type ON violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_compliance_session ON compliance_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_compliance_timestamp ON compliance_snapshots(timestamp);
