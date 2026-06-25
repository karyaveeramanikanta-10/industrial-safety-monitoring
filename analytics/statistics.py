"""
Statistical analysis for safety monitoring data.

Provides calculations for compliance scores, violation summaries,
trend analysis, and risk assessment.
"""

import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger('safety_monitor')


class SafetyStatistics:
    """Calculate and track safety monitoring statistics.

    Provides real-time and historical statistical analysis
    for the safety monitoring dashboard.
    """

    def __init__(self, db_manager=None):
        """Initialize statistics engine.

        Args:
            db_manager: DatabaseManager instance for historical queries.
        """
        self.db_manager = db_manager
        self._cache: Dict = {}
        self._cache_time: Dict = {}
        self._cache_ttl = 5  # seconds

    def get_realtime_stats(self, current_workers: int = 0,
                            compliant_workers: int = 0,
                            violations: int = 0,
                            fps: float = 0.0) -> Dict:
        """Get real-time monitoring statistics.

        Args:
            current_workers: Number of currently detected workers.
            compliant_workers: Number of compliant workers.
            violations: Total violation count.
            fps: Current frames per second.

        Returns:
            Dict with all real-time stats.
        """
        return {
            'total_workers': current_workers,
            'compliant_workers': compliant_workers,
            'total_violations': violations,
            'fps': fps,
            'compliance_rate': (
                (compliant_workers / max(current_workers, 1)) * 100
            ),
            'timestamp': datetime.now().isoformat(),
        }

    def get_violation_summary(self, hours: int = 24) -> Dict:
        """Get violation summary for the last N hours.

        Args:
            hours: Number of hours to look back.

        Returns:
            Summary dict with counts and breakdowns.
        """
        if not self.db_manager:
            return {}

        by_type = self.db_manager.get_violations_by_type()
        trends = self.db_manager.get_violation_trends(hours)
        total = sum(by_type.values()) if by_type else 0

        return {
            'total_violations': total,
            'by_type': by_type,
            'hourly_trends': trends,
            'most_common': max(by_type, key=by_type.get) if by_type else None,
            'period_hours': hours,
        }

    def get_worker_compliance_scores(self) -> List[Dict]:
        """Calculate compliance score for each worker.

        Score = 100 - (violations / frames_tracked * 100), clamped [0, 100].

        Returns:
            List of worker dicts with compliance_score.
        """
        if not self.db_manager:
            return []

        workers = self.db_manager.get_worker_stats()
        results = []

        for w in workers:
            frames = w.get('total_frames_tracked', 1) or 1
            violations = w.get('total_violations', 0)
            score = max(0, 100 - (violations / frames * 100))
            results.append({
                'worker_id': w['worker_id'],
                'compliance_score': round(score, 1),
                'total_violations': violations,
                'frames_tracked': frames,
            })

        return sorted(results, key=lambda x: x['compliance_score'])

    def get_peak_violation_hours(self) -> List[Dict]:
        """Identify hours with most violations.

        Returns:
            List of {'hour': str, 'count': int} sorted by count desc.
        """
        if not self.db_manager:
            return []

        trends = self.db_manager.get_violation_trends(hours=168)  # 7 days
        if not trends:
            return []

        sorted_hours = sorted(
            trends.items(), key=lambda x: x[1], reverse=True
        )
        return [{'hour': h, 'count': c} for h, c in sorted_hours[:10]]

    def get_trend_data(self, metric: str = 'violations',
                        days: int = 7) -> Dict:
        """Get trend data for dashboard charts.

        Args:
            metric: Metric to trend ('violations', 'compliance').
            days: Number of days to look back.

        Returns:
            Dict with labels and values for charting.
        """
        if not self.db_manager:
            return {'labels': [], 'values': []}

        if metric == 'violations':
            trends = self.db_manager.get_violation_trends(hours=days * 24)
            return {
                'labels': list(trends.keys()),
                'values': list(trends.values()),
            }
        elif metric == 'compliance':
            history = self.db_manager.get_compliance_history(hours=days * 24)
            return {
                'labels': [h['timestamp'] for h in history],
                'values': [h['compliance_rate'] for h in history],
            }

        return {'labels': [], 'values': []}

    def calculate_risk_score(self, worker_id: int) -> float:
        """Calculate risk score for a worker (0=low, 100=high).

        Based on recent violation frequency and recency.

        Args:
            worker_id: Worker tracking ID.

        Returns:
            Risk score (0-100).
        """
        if not self.db_manager:
            return 0.0

        violations = self.db_manager.get_worker_violations(worker_id)
        if not violations:
            return 0.0

        # Factor 1: Total violations (more = higher risk)
        count_score = min(len(violations) * 10, 50)

        # Factor 2: Recency (recent violations = higher risk)
        latest = violations[0].get('timestamp', '')
        try:
            latest_dt = datetime.fromisoformat(latest)
            hours_ago = (datetime.now() - latest_dt).total_seconds() / 3600
            recency_score = max(0, 50 - hours_ago * 2)
        except (ValueError, TypeError):
            recency_score = 25

        return min(100, count_score + recency_score)

    def generate_daily_summary(self) -> Dict:
        """Generate end-of-day summary statistics.

        Returns:
            Comprehensive daily summary dict.
        """
        if not self.db_manager:
            return {}

        counts = self.db_manager.get_total_counts()
        by_type = self.db_manager.get_violations_by_type()
        compliance = self.db_manager.get_compliance_rate()

        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_workers_tracked': counts.get('total_workers', 0),
            'total_violations': counts.get('total_violations', 0),
            'average_compliance_rate': compliance,
            'violations_by_type': by_type,
            'generated_at': datetime.now().isoformat(),
        }
