"""
AWS CloudWatch monitoring integration for RAG system.
Publishes custom metrics (latency, error rate, cache hit rate,
document count) to CloudWatch for dashboarding and alarming.
Gracefully degrades to local-only metrics when AWS is unavailable.
"""

import time
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger("rag_monitoring")


class LocalMonitor:
    """
    In-memory metrics store used as a fallback when AWS CloudWatch
    credentials are not available.

    Provides the same public interface as ``CloudWatchMonitor`` so the
    two can be used interchangeably.
    """

    def __init__(self, namespace: str = "RAGSystem"):
        self.namespace = namespace
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._errors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._cache_hits = 0
        self._cache_misses = 0
        self._retrieval_quality: List[Dict[str, Any]] = []
        self._custom_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Core recording methods
    # ------------------------------------------------------------------

    def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "None",
        dimensions: Optional[List[Dict]] = None,
    ) -> None:
        """Store a single metric data-point locally."""
        self._custom_metrics[metric_name].append(
            {
                "value": value,
                "unit": unit,
                "dimensions": dimensions or [],
                "timestamp": time.time(),
            }
        )

    def record_latency(self, endpoint: str, latency_ms: float) -> None:
        """Record request latency for *endpoint*."""
        self._latencies[endpoint].append(latency_ms)
        self.put_metric(
            "RequestLatency",
            latency_ms,
            unit="Milliseconds",
            dimensions=[{"Name": "Endpoint", "Value": endpoint}],
        )

    def record_error(self, endpoint: str, error_type: str) -> None:
        """Record an error occurrence."""
        self._errors[endpoint][error_type] += 1
        self.put_metric(
            "ErrorCount",
            1,
            unit="Count",
            dimensions=[
                {"Name": "Endpoint", "Value": endpoint},
                {"Name": "ErrorType", "Value": error_type},
            ],
        )

    def record_cache_hit(self, hit: bool) -> None:
        """Track a cache hit or miss."""
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        self.put_metric("CacheHit", 1.0 if hit else 0.0, unit="Count")

    def record_retrieval_quality(self, confidence: str, num_sources: int) -> None:
        """Track retrieval quality for a single query."""
        self._retrieval_quality.append(
            {
                "confidence": confidence,
                "num_sources": num_sources,
                "timestamp": time.time(),
            }
        )
        self.put_metric(
            "RetrievalSources",
            float(num_sources),
            unit="Count",
            dimensions=[{"Name": "Confidence", "Value": confidence}],
        )

    def publish_system_health(self, metrics: Dict) -> None:
        """Store batch system health metrics."""
        for name, value in metrics.items():
            self.put_metric(name, float(value))

    def setup_alarms(self, sns_topic_arn: Optional[str] = None) -> None:
        """No-op for local monitor -- alarms require CloudWatch."""
        logger.info("LocalMonitor: alarms are not supported locally (no-op).")

    def get_dashboard_url(self) -> str:
        """Return a placeholder URL."""
        return "http://localhost/metrics  (local-only, no CloudWatch dashboard)"

    # ------------------------------------------------------------------
    # Retrieval / reporting
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return a summary of all locally collected metrics."""
        total_cache = self._cache_hits + self._cache_misses

        # Per-endpoint latency summaries
        latency_summary: Dict[str, Dict[str, float]] = {}
        for endpoint, values in self._latencies.items():
            if values:
                sorted_vals = sorted(values)
                latency_summary[endpoint] = {
                    "count": len(values),
                    "mean_ms": round(sum(values) / len(values), 2),
                    "p50_ms": round(sorted_vals[len(sorted_vals) // 2], 2),
                    "p99_ms": round(sorted_vals[int(len(sorted_vals) * 0.99)], 2),
                    "max_ms": round(max(values), 2),
                }

        # Error totals
        error_summary: Dict[str, Dict[str, int]] = {}
        for endpoint, types in self._errors.items():
            error_summary[endpoint] = dict(types)

        return {
            "namespace": self.namespace,
            "backend": "local",
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "latency": latency_summary,
            "errors": error_summary,
            "cache": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": (round(self._cache_hits / total_cache, 4) if total_cache > 0 else 0.0),
            },
            "retrieval_quality": {
                "total_queries": len(self._retrieval_quality),
                "recent": self._retrieval_quality[-10:],
            },
        }


class CloudWatchMonitor:
    """
    Publishes custom metrics to AWS CloudWatch.

    When AWS credentials are missing or boto3 is not installed the monitor
    automatically falls back to a ``LocalMonitor`` so callers never need
    to handle AWS availability themselves.
    """

    def __init__(
        self,
        namespace: str = "RAGSystem",
        region: str = "us-east-1",
        enabled: bool = True,
    ):
        self.namespace = namespace
        self.region = region
        self.enabled = enabled
        self._client = None
        self._fallback = LocalMonitor(namespace=namespace)

        if not self.enabled:
            logger.info("CloudWatch monitoring explicitly disabled; using local metrics.")
            return

        try:
            import boto3

            self._client = boto3.client(
                "cloudwatch",
                region_name=self.region,
            )
            # Lightweight validation -- list_metrics with a limit of 1
            self._client.list_metrics(Namespace=self.namespace, MaxRecords=1)
            logger.info(
                "CloudWatch client initialised (namespace=%s, region=%s)",
                self.namespace,
                self.region,
            )
        except ImportError:
            logger.warning("boto3 is not installed; falling back to local metrics.")
            self._client = None
        except Exception as exc:
            logger.warning("CloudWatch unavailable (%s); falling back to local metrics.", exc)
            self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_available(self) -> bool:
        return self._client is not None and self.enabled

    def _safe_put(self, metric_data: List[Dict]) -> None:
        """Put metric data to CloudWatch, logging any errors."""
        if not self._is_available():
            return
        try:
            self._client.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data,
            )
        except Exception as exc:
            logger.warning("Failed to publish metric to CloudWatch: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "None",
        dimensions: Optional[List[Dict]] = None,
    ) -> None:
        """Publish a single metric to CloudWatch."""
        # Always record locally for get_metrics()
        self._fallback.put_metric(metric_name, value, unit, dimensions)

        if not self._is_available():
            return

        datum: Dict[str, Any] = {
            "MetricName": metric_name,
            "Value": value,
            "Unit": unit,
        }
        if dimensions:
            datum["Dimensions"] = [{"Name": d["Name"], "Value": d["Value"]} for d in dimensions]

        self._safe_put([datum])

    def record_latency(self, endpoint: str, latency_ms: float) -> None:
        """Record request latency for *endpoint*."""
        self._fallback.record_latency(endpoint, latency_ms)

        if not self._is_available():
            return

        self._safe_put(
            [
                {
                    "MetricName": "RequestLatency",
                    "Value": latency_ms,
                    "Unit": "Milliseconds",
                    "Dimensions": [{"Name": "Endpoint", "Value": endpoint}],
                }
            ]
        )

    def record_error(self, endpoint: str, error_type: str) -> None:
        """Record error occurrence."""
        self._fallback.record_error(endpoint, error_type)

        if not self._is_available():
            return

        self._safe_put(
            [
                {
                    "MetricName": "ErrorCount",
                    "Value": 1,
                    "Unit": "Count",
                    "Dimensions": [
                        {"Name": "Endpoint", "Value": endpoint},
                        {"Name": "ErrorType", "Value": error_type},
                    ],
                }
            ]
        )

    def record_cache_hit(self, hit: bool) -> None:
        """Track cache hit/miss ratio."""
        self._fallback.record_cache_hit(hit)

        if not self._is_available():
            return

        self._safe_put(
            [
                {
                    "MetricName": "CacheHit",
                    "Value": 1.0 if hit else 0.0,
                    "Unit": "Count",
                },
            ]
        )

    def record_retrieval_quality(self, confidence: str, num_sources: int) -> None:
        """Track retrieval quality metrics."""
        self._fallback.record_retrieval_quality(confidence, num_sources)

        if not self._is_available():
            return

        self._safe_put(
            [
                {
                    "MetricName": "RetrievalSources",
                    "Value": float(num_sources),
                    "Unit": "Count",
                    "Dimensions": [{"Name": "Confidence", "Value": confidence}],
                },
                {
                    "MetricName": "RetrievalConfidence",
                    "Value": 1.0,
                    "Unit": "Count",
                    "Dimensions": [{"Name": "Level", "Value": confidence}],
                },
            ]
        )

    def publish_system_health(self, metrics: Dict) -> None:
        """
        Publish a batch of system-health metrics.

        Expected keys include ``uptime_seconds``, ``total_documents``,
        ``error_rate``, etc.
        """
        self._fallback.publish_system_health(metrics)

        if not self._is_available():
            return

        metric_data = []
        unit_map = {
            "uptime_seconds": "Seconds",
            "total_documents": "Count",
            "error_rate": "Percent",
            "total_requests": "Count",
            "error_count": "Count",
            "average_latency_ms": "Milliseconds",
        }

        for name, value in metrics.items():
            try:
                metric_data.append(
                    {
                        "MetricName": name,
                        "Value": float(value),
                        "Unit": unit_map.get(name, "None"),
                    }
                )
            except (TypeError, ValueError):
                continue

        if metric_data:
            self._safe_put(metric_data)

    def setup_alarms(self, sns_topic_arn: Optional[str] = None) -> None:
        """
        Create CloudWatch alarms for critical operational thresholds.

        Alarms created:
        - **HighErrorRate** -- error rate > 5 % over 5 min
        - **HighLatencyP99** -- p99 latency > 2 000 ms over 5 min
        - **LowAvailability** -- fewer than 1 healthy request per minute
        """
        if not self._is_available():
            logger.info("CloudWatch unavailable; skipping alarm setup.")
            return

        actions = [sns_topic_arn] if sns_topic_arn else []

        alarm_configs = [
            {
                "AlarmName": f"{self.namespace}-HighErrorRate",
                "AlarmDescription": "Error rate exceeds 5% over 5 minutes",
                "MetricName": "ErrorCount",
                "Namespace": self.namespace,
                "Statistic": "Sum",
                "Period": 300,
                "EvaluationPeriods": 1,
                "Threshold": 5.0,
                "ComparisonOperator": "GreaterThanThreshold",
                "TreatMissingData": "notBreaching",
            },
            {
                "AlarmName": f"{self.namespace}-HighLatencyP99",
                "AlarmDescription": "p99 latency exceeds 2000ms over 5 minutes",
                "MetricName": "RequestLatency",
                "Namespace": self.namespace,
                "ExtendedStatistic": "p99",
                "Period": 300,
                "EvaluationPeriods": 1,
                "Threshold": 2000.0,
                "ComparisonOperator": "GreaterThanThreshold",
                "TreatMissingData": "notBreaching",
            },
            {
                "AlarmName": f"{self.namespace}-LowAvailability",
                "AlarmDescription": "Fewer than 1 request per minute (service may be down)",
                "MetricName": "RequestLatency",
                "Namespace": self.namespace,
                "Statistic": "SampleCount",
                "Period": 300,
                "EvaluationPeriods": 1,
                "Threshold": 1.0,
                "ComparisonOperator": "LessThanThreshold",
                "TreatMissingData": "breaching",
            },
        ]

        for cfg in alarm_configs:
            try:
                put_kwargs = {
                    "AlarmName": cfg["AlarmName"],
                    "AlarmDescription": cfg["AlarmDescription"],
                    "Namespace": cfg["Namespace"],
                    "MetricName": cfg["MetricName"],
                    "Period": cfg["Period"],
                    "EvaluationPeriods": cfg["EvaluationPeriods"],
                    "Threshold": cfg["Threshold"],
                    "ComparisonOperator": cfg["ComparisonOperator"],
                    "TreatMissingData": cfg["TreatMissingData"],
                }

                # Use either Statistic or ExtendedStatistic, not both
                if "ExtendedStatistic" in cfg:
                    put_kwargs["ExtendedStatistic"] = cfg["ExtendedStatistic"]
                else:
                    put_kwargs["Statistic"] = cfg["Statistic"]

                if actions:
                    put_kwargs["AlarmActions"] = actions
                    put_kwargs["OKActions"] = actions

                self._client.put_metric_alarm(**put_kwargs)
                logger.info("Created CloudWatch alarm: %s", cfg["AlarmName"])
            except Exception as exc:
                logger.warning("Failed to create alarm %s: %s", cfg["AlarmName"], exc)

    def get_dashboard_url(self) -> str:
        """Return the CloudWatch dashboard URL for this namespace."""
        if not self._is_available():
            return self._fallback.get_dashboard_url()

        return (
            f"https://{self.region}.console.aws.amazon.com/cloudwatch/home"
            f"?region={self.region}#metricsV2:"
            f"graph=~();namespace=~'{self.namespace}'"
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return local metric summaries.

        Even when CloudWatch is available the local fallback always
        collects data, so this method works in all modes.
        """
        summary = self._fallback.get_metrics()
        summary["backend"] = "cloudwatch" if self._is_available() else "local"
        if self._is_available():
            summary["dashboard_url"] = self.get_dashboard_url()
        return summary
