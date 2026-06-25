"""
Report generation for safety monitoring.

Generates CSV and JSON reports from violation data with
filtering by date range, worker, or session.
"""

import os
import csv
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger('safety_monitor')


class ReportGenerator:
    """Generate safety monitoring reports in CSV and JSON formats.

    Usage:
        generator = ReportGenerator(db_manager)
        csv_path = generator.generate_csv_report()
        json_path = generator.generate_json_report()
    """

    def __init__(self, db_manager, output_dir: str = 'data/violation_logs'):
        """Initialize report generator.

        Args:
            db_manager: DatabaseManager instance.
            output_dir: Directory for report files.
        """
        self.db_manager = db_manager
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_csv_report(self, start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             output_file: Optional[str] = None) -> str:
        """Generate CSV violation report.

        Args:
            start_date: ISO date string filter (inclusive).
            end_date: ISO date string filter (inclusive).
            output_file: Custom output filename.

        Returns:
            Path to generated CSV file.
        """
        if not output_file:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.output_dir, f'violations_{ts}.csv'
            )

        violations = self.db_manager.get_recent_violations(limit=10000)

        # Apply date filters
        if start_date or end_date:
            filtered = []
            for v in violations:
                ts = v.get('timestamp', '')
                if start_date and ts < start_date:
                    continue
                if end_date and ts > end_date:
                    continue
                filtered.append(v)
            violations = filtered

        if not violations:
            logger.warning("No violations to report")
            # Write empty CSV with headers
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'ID', 'Worker ID', 'Violation Type',
                    'Timestamp', 'Frame', 'Confidence'
                ])
            return output_file

        headers = list(violations[0].keys())
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(violations)

        logger.info(
            f"CSV report generated: {output_file} "
            f"({len(violations)} violations)"
        )
        return output_file

    def generate_json_report(self, start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              output_file: Optional[str] = None) -> str:
        """Generate JSON report with full statistics.

        Args:
            start_date: ISO date string filter.
            end_date: ISO date string filter.
            output_file: Custom output filename.

        Returns:
            Path to generated JSON file.
        """
        if not output_file:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(
                self.output_dir, f'report_{ts}.json'
            )

        violations = self.db_manager.get_recent_violations(limit=10000)
        by_type = self.db_manager.get_violations_by_type()
        workers = self.db_manager.get_worker_stats()
        sessions = self.db_manager.get_all_sessions()

        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_violations': len(violations),
                'violations_by_type': by_type,
                'total_workers': len(workers),
                'total_sessions': len(sessions),
                'compliance_rate': self.db_manager.get_compliance_rate(),
            },
            'workers': workers,
            'sessions': sessions,
            'violations': violations,
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"JSON report generated: {output_file}")
        return output_file

    def generate_worker_report(self, worker_id: int,
                                output_file: Optional[str] = None) -> str:
        """Generate report for a specific worker.

        Args:
            worker_id: Worker tracking ID.
            output_file: Custom output filename.

        Returns:
            Path to generated report file.
        """
        if not output_file:
            output_file = os.path.join(
                self.output_dir, f'worker_{worker_id}_report.json'
            )

        violations = self.db_manager.get_worker_violations(worker_id)

        report = {
            'worker_id': worker_id,
            'generated_at': datetime.now().isoformat(),
            'total_violations': len(violations),
            'violations': violations,
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Worker {worker_id} report: {output_file}")
        return output_file

    def generate_session_report(self, session_id: int,
                                 output_file: Optional[str] = None) -> str:
        """Generate report for a monitoring session.

        Args:
            session_id: Session ID.
            output_file: Custom output filename.

        Returns:
            Path to generated report file.
        """
        if not output_file:
            output_file = os.path.join(
                self.output_dir, f'session_{session_id}_report.json'
            )

        stats = self.db_manager.get_session_stats(session_id)

        report = {
            'session_id': session_id,
            'generated_at': datetime.now().isoformat(),
            'session_stats': stats,
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Session {session_id} report: {output_file}")
        return output_file

    def export_violation_log(self,
                              output_file: Optional[str] = None) -> str:
        """Export full violation log as CSV.

        Args:
            output_file: Custom output path.

        Returns:
            Path to exported CSV.
        """
        return self.generate_csv_report(output_file=output_file)
