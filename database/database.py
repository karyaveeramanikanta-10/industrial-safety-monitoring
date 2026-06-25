"""
Database manager for Industrial Safety Monitoring System.

Provides thread-safe MongoDB Atlas operations for storing and querying
violation logs, worker statistics, sessions, and compliance data.

Collections:
    - workers: Tracks each detected and tracked worker
    - violations: Records each PPE violation event
    - sessions: One document per monitoring run
    - compliance_snapshots: Periodic compliance rate recordings
"""

import os
import threading
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

from pymongo import MongoClient, DESCENDING, ASCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger('safety_monitor')


class DatabaseManager:
    """Thread-safe MongoDB Atlas database manager.

    Manages all database operations including violation logging,
    worker tracking, session management, and compliance snapshots.
    Uses MongoDB Atlas as the backend.
    """

    def __init__(self, mongo_uri: str = None, db_name: str = 'safety_monitoring'):
        """Initialize MongoDB connection.

        Args:
            mongo_uri: MongoDB connection URI. If None, reads from
                       MONGO_URI environment variable.
            db_name: Name of the MongoDB database.
        """
        self._lock = threading.Lock()

        # Resolve connection URI: parameter > env var > fallback
        self.mongo_uri = mongo_uri or os.environ.get(
            'MONGO_URI',
            'mongodb://localhost:27017/'
        )
        self.db_name = db_name

        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )
            # Verify connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self._init_collections()
            logger.info(f"MongoDB connected: {self.db_name}")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise RuntimeError(
                f"Cannot connect to MongoDB Atlas. "
                f"Check your MONGO_URI environment variable. Error: {e}"
            )

    def _init_collections(self):
        """Initialize collections and create indexes."""
        self.workers = self.db['workers']
        self.violations = self.db['violations']
        self.sessions = self.db['sessions']
        self.compliance_snapshots = self.db['compliance_snapshots']

        # Create indexes for query performance
        self.violations.create_index([('worker_id', ASCENDING)])
        self.violations.create_index([('timestamp', DESCENDING)])
        self.violations.create_index([('session_id', ASCENDING)])
        self.violations.create_index([('violation_type', ASCENDING)])
        self.compliance_snapshots.create_index([('session_id', ASCENDING)])
        self.compliance_snapshots.create_index([('timestamp', DESCENDING)])
        self.workers.create_index([('worker_id', ASCENDING)], unique=True)
        self.sessions.create_index([('session_id', ASCENDING)], unique=True)

        logger.debug("MongoDB indexes created")

    def _next_session_id(self) -> int:
        """Generate auto-incrementing session ID."""
        counters = self.db['counters']
        result = counters.find_one_and_update(
            {'_id': 'session_id'},
            {'$inc': {'seq': 1}},
            upsert=True,
            return_document=True
        )
        return result['seq']

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def create_session(self, video_source: str) -> int:
        """Create a new monitoring session. Returns session_id."""
        with self._lock:
            session_id = self._next_session_id()
            self.sessions.insert_one({
                'session_id': session_id,
                'start_time': datetime.utcnow(),
                'end_time': None,
                'video_source': video_source,
                'total_frames': 0,
                'total_detections': 0,
                'total_violations': 0,
            })
            logger.info(f"Created session {session_id} for source: {video_source}")
            return session_id

    def end_session(self, session_id: int, total_frames: int = 0,
                    total_detections: int = 0, total_violations: int = 0):
        """End a monitoring session with final statistics."""
        with self._lock:
            self.sessions.update_one(
                {'session_id': session_id},
                {'$set': {
                    'end_time': datetime.utcnow(),
                    'total_frames': total_frames,
                    'total_detections': total_detections,
                    'total_violations': total_violations,
                }}
            )
            logger.info(f"Ended session {session_id}")

    def get_session_stats(self, session_id: int) -> Optional[Dict]:
        """Get statistics for a specific session."""
        doc = self.sessions.find_one(
            {'session_id': session_id}, {'_id': 0}
        )
        if doc:
            # Convert datetime objects to ISO strings for compatibility
            for key in ('start_time', 'end_time'):
                if doc.get(key) and isinstance(doc[key], datetime):
                    doc[key] = doc[key].isoformat()
        return doc

    def get_all_sessions(self) -> List[Dict]:
        """Get all monitoring sessions."""
        docs = list(self.sessions.find(
            {}, {'_id': 0}
        ).sort('start_time', DESCENDING))
        for doc in docs:
            for key in ('start_time', 'end_time'):
                if doc.get(key) and isinstance(doc[key], datetime):
                    doc[key] = doc[key].isoformat()
        return docs

    # -------------------------------------------------------------------------
    # Worker Management
    # -------------------------------------------------------------------------

    def register_worker(self, worker_id: int):
        """Register a new worker or update existing worker's last_seen."""
        with self._lock:
            now = datetime.utcnow()
            self.workers.update_one(
                {'worker_id': worker_id},
                {
                    '$set': {'last_seen': now},
                    '$setOnInsert': {
                        'worker_id': worker_id,
                        'first_seen': now,
                        'total_violations': 0,
                        'total_frames_tracked': 0,
                    }
                },
                upsert=True
            )

    def update_worker(self, worker_id: int, frames_tracked: int = 1):
        """Update worker tracking statistics."""
        with self._lock:
            self.workers.update_one(
                {'worker_id': worker_id},
                {
                    '$set': {'last_seen': datetime.utcnow()},
                    '$inc': {'total_frames_tracked': frames_tracked}
                }
            )

    def get_worker_stats(self) -> List[Dict]:
        """Get statistics for all workers with violation counts."""
        pipeline = [
            {
                '$lookup': {
                    'from': 'violations',
                    'localField': 'worker_id',
                    'foreignField': 'worker_id',
                    'as': 'violation_docs'
                }
            },
            {
                '$addFields': {
                    'violation_count': {'$size': '$violation_docs'},
                    'violation_types': {
                        '$reduce': {
                            'input': {'$setUnion': '$violation_docs.violation_type'},
                            'initialValue': '',
                            'in': {
                                '$cond': [
                                    {'$eq': ['$$value', '']},
                                    '$$this',
                                    {'$concat': ['$$value', ',', '$$this']}
                                ]
                            }
                        }
                    }
                }
            },
            {'$project': {'_id': 0, 'violation_docs': 0}},
            {'$sort': {'worker_id': 1}}
        ]
        docs = list(self.workers.aggregate(pipeline))
        for doc in docs:
            for key in ('first_seen', 'last_seen'):
                if doc.get(key) and isinstance(doc[key], datetime):
                    doc[key] = doc[key].isoformat()
        return docs

    # -------------------------------------------------------------------------
    # Violation Logging
    # -------------------------------------------------------------------------

    def log_violation(self, worker_id: int, violation_type: str,
                      frame_number: int, confidence: float = 0.0,
                      snapshot_path: str = None, session_id: int = None):
        """Log a PPE violation event."""
        with self._lock:
            now = datetime.utcnow()

            # Ensure worker exists (upsert)
            self.workers.update_one(
                {'worker_id': worker_id},
                {
                    '$set': {'last_seen': now},
                    '$setOnInsert': {
                        'worker_id': worker_id,
                        'first_seen': now,
                        'total_violations': 0,
                        'total_frames_tracked': 0,
                    }
                },
                upsert=True
            )

            # Insert violation document
            self.violations.insert_one({
                'worker_id': worker_id,
                'violation_type': violation_type,
                'timestamp': now,
                'frame_number': frame_number,
                'confidence': confidence,
                'snapshot_path': snapshot_path,
                'resolved': False,
                'session_id': session_id,
            })

            # Increment worker violation count
            self.workers.update_one(
                {'worker_id': worker_id},
                {'$inc': {'total_violations': 1}}
            )

            logger.debug(
                f"Violation logged: Worker {worker_id}, Type: {violation_type}, "
                f"Frame: {frame_number}"
            )

    def get_worker_violations(self, worker_id: int) -> List[Dict]:
        """Get all violations for a specific worker."""
        docs = list(self.violations.find(
            {'worker_id': worker_id},
            {'_id': 0}
        ).sort('timestamp', DESCENDING))
        for doc in docs:
            if doc.get('timestamp') and isinstance(doc['timestamp'], datetime):
                doc['timestamp'] = doc['timestamp'].isoformat()
        return docs

    def get_recent_violations(self, limit: int = 50) -> List[Dict]:
        """Get most recent violations across all workers."""
        pipeline = [
            {'$sort': {'timestamp': -1}},
            {'$limit': limit},
            {
                '$lookup': {
                    'from': 'workers',
                    'localField': 'worker_id',
                    'foreignField': 'worker_id',
                    'as': 'worker_info'
                }
            },
            {
                '$addFields': {
                    'worker_total': {
                        '$ifNull': [
                            {'$arrayElemAt': ['$worker_info.total_violations', 0]},
                            0
                        ]
                    },
                    'first_seen': {
                        '$arrayElemAt': ['$worker_info.first_seen', 0]
                    }
                }
            },
            {'$project': {'_id': 0, 'worker_info': 0}}
        ]
        docs = list(self.violations.aggregate(pipeline))
        for doc in docs:
            for key in ('timestamp', 'first_seen'):
                if doc.get(key) and isinstance(doc[key], datetime):
                    doc[key] = doc[key].isoformat()
        return docs

    def get_violations_by_type(self) -> Dict[str, int]:
        """Get violation counts grouped by type."""
        pipeline = [
            {'$group': {'_id': '$violation_type', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        result = list(self.violations.aggregate(pipeline))
        return {doc['_id']: doc['count'] for doc in result}

    def get_violation_trends(self, hours: int = 24) -> Dict[str, int]:
        """Get violation counts grouped by hour for the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        pipeline = [
            {'$match': {'timestamp': {'$gte': cutoff}}},
            {
                '$group': {
                    '_id': {
                        '$dateToString': {
                            'format': '%Y-%m-%d %H:00',
                            'date': '$timestamp'
                        }
                    },
                    'count': {'$sum': 1}
                }
            },
            {'$sort': {'_id': 1}}
        ]
        result = list(self.violations.aggregate(pipeline))
        return {doc['_id']: doc['count'] for doc in result}

    # -------------------------------------------------------------------------
    # Compliance
    # -------------------------------------------------------------------------

    def get_compliance_rate(self, session_id: int = None) -> float:
        """Calculate overall compliance rate."""
        match = {}
        if session_id:
            match = {'session_id': session_id}

        pipeline = [
            {'$match': match} if match else {'$match': {}},
            {'$group': {'_id': None, 'avg_rate': {'$avg': '$compliance_rate'}}}
        ]
        result = list(self.compliance_snapshots.aggregate(pipeline))
        if result and result[0].get('avg_rate') is not None:
            return result[0]['avg_rate']
        return 0.0

    def save_compliance_snapshot(self, session_id: int, total_workers: int,
                                 compliant_workers: int, compliance_rate: float):
        """Save a compliance rate snapshot."""
        with self._lock:
            self.compliance_snapshots.insert_one({
                'session_id': session_id,
                'timestamp': datetime.utcnow(),
                'total_workers': total_workers,
                'compliant_workers': compliant_workers,
                'compliance_rate': compliance_rate,
            })

    def get_compliance_history(self, hours: int = 24) -> List[Dict]:
        """Get compliance snapshots for the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        docs = list(self.compliance_snapshots.find(
            {'timestamp': {'$gte': cutoff}},
            {'_id': 0}
        ).sort('timestamp', ASCENDING))
        for doc in docs:
            if doc.get('timestamp') and isinstance(doc['timestamp'], datetime):
                doc['timestamp'] = doc['timestamp'].isoformat()
        return docs

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    def clear_old_violations(self, days: int = 30):
        """Delete violations older than N days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self._lock:
            result = self.violations.delete_many(
                {'timestamp': {'$lt': cutoff}}
            )
            logger.info(
                f"Cleared {result.deleted_count} old violations (>{days} days)"
            )

    def get_total_counts(self) -> Dict[str, int]:
        """Get total counts for dashboard display."""
        return {
            'total_workers': self.workers.count_documents({}),
            'total_violations': self.violations.count_documents({}),
            'total_sessions': self.sessions.count_documents({}),
        }

    def close(self):
        """Close the MongoDB connection."""
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
