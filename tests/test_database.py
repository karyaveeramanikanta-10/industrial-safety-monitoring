"""Unit tests for MongoDB database module."""

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Tests for MongoDB DatabaseManager class.

    Uses the MONGO_URI environment variable for connection.
    Tests use a dedicated test database that is cleaned up after each run.
    """

    TEST_DB_NAME = 'safety_monitoring_test'

    @classmethod
    def setUpClass(cls):
        """Connect to MongoDB once for all tests."""
        mongo_uri = os.environ.get('MONGO_URI')
        if not mongo_uri:
            raise unittest.SkipTest(
                "MONGO_URI environment variable not set. "
                "Set it to run database tests."
            )
        cls.db = DatabaseManager(
            mongo_uri=mongo_uri,
            db_name=cls.TEST_DB_NAME
        )

    def setUp(self):
        """Clear test collections before each test."""
        self.db.workers.delete_many({})
        self.db.violations.delete_many({})
        self.db.sessions.delete_many({})
        self.db.compliance_snapshots.delete_many({})
        # Reset session counter
        self.db.db['counters'].delete_many({})

    @classmethod
    def tearDownClass(cls):
        """Drop the test database after all tests."""
        if hasattr(cls, 'db') and cls.db:
            cls.db.client.drop_database(cls.TEST_DB_NAME)
            cls.db.close()

    def test_initialization(self):
        self.assertIsNotNone(self.db)
        self.assertIsNotNone(self.db.client)

    def test_create_session(self):
        sid = self.db.create_session('test_video.mp4')
        self.assertIsNotNone(sid)
        self.assertGreater(sid, 0)

    def test_end_session(self):
        sid = self.db.create_session('test')
        self.db.end_session(sid, 100, 50, 5)
        stats = self.db.get_session_stats(sid)
        self.assertEqual(stats['total_frames'], 100)
        self.assertEqual(stats['total_violations'], 5)

    def test_register_worker(self):
        self.db.register_worker(1)
        stats = self.db.get_worker_stats()
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]['worker_id'], 1)

    def test_log_violation(self):
        sid = self.db.create_session('test')
        self.db.register_worker(1)
        self.db.log_violation(1, 'helmet', 100, 0.9, session_id=sid)
        violations = self.db.get_worker_violations(1)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['violation_type'], 'helmet')

    def test_multiple_violations(self):
        sid = self.db.create_session('test')
        self.db.register_worker(1)
        self.db.log_violation(1, 'helmet', 1, session_id=sid)
        self.db.log_violation(1, 'vest', 2, session_id=sid)
        self.db.log_violation(1, 'helmet', 3, session_id=sid)
        violations = self.db.get_worker_violations(1)
        self.assertEqual(len(violations), 3)

    def test_violations_by_type(self):
        sid = self.db.create_session('test')
        self.db.register_worker(1)
        self.db.log_violation(1, 'helmet', 1, session_id=sid)
        self.db.log_violation(1, 'vest', 2, session_id=sid)
        self.db.log_violation(1, 'helmet', 3, session_id=sid)
        by_type = self.db.get_violations_by_type()
        self.assertEqual(by_type.get('helmet'), 2)
        self.assertEqual(by_type.get('vest'), 1)

    def test_recent_violations(self):
        sid = self.db.create_session('test')
        self.db.register_worker(1)
        for i in range(10):
            self.db.log_violation(1, 'helmet', i, session_id=sid)
        recent = self.db.get_recent_violations(5)
        self.assertEqual(len(recent), 5)

    def test_worker_stats(self):
        self.db.register_worker(1)
        self.db.register_worker(2)
        stats = self.db.get_worker_stats()
        self.assertEqual(len(stats), 2)

    def test_compliance_snapshot(self):
        sid = self.db.create_session('test')
        self.db.save_compliance_snapshot(sid, 10, 8, 80.0)
        rate = self.db.get_compliance_rate(sid)
        self.assertAlmostEqual(rate, 80.0)

    def test_total_counts(self):
        self.db.register_worker(1)
        self.db.register_worker(2)
        counts = self.db.get_total_counts()
        self.assertEqual(counts['total_workers'], 2)

    def test_get_all_sessions(self):
        self.db.create_session('test1')
        self.db.create_session('test2')
        sessions = self.db.get_all_sessions()
        self.assertEqual(len(sessions), 2)

    def test_update_worker(self):
        self.db.register_worker(1)
        self.db.update_worker(1, frames_tracked=10)
        stats = self.db.get_worker_stats()
        self.assertEqual(stats[0]['total_frames_tracked'], 10)

    def test_violation_trends(self):
        sid = self.db.create_session('test')
        self.db.register_worker(1)
        self.db.log_violation(1, 'helmet', 1, session_id=sid)
        trends = self.db.get_violation_trends(hours=1)
        self.assertGreater(len(trends), 0)

    def test_compliance_history(self):
        sid = self.db.create_session('test')
        self.db.save_compliance_snapshot(sid, 5, 4, 80.0)
        self.db.save_compliance_snapshot(sid, 5, 5, 100.0)
        history = self.db.get_compliance_history(hours=1)
        self.assertEqual(len(history), 2)


if __name__ == '__main__':
    unittest.main()
