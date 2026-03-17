# Data Quality & Model Monitoring

This project is a personal RAG system, so it does not ingest live wearable data by default. If you extend it into a real-time pipeline, consider adding the following:

## Data Validation (Pre-Ingest)

Recommended tools:

- **Pydantic**: Schema-level validation for API requests and data payloads.
- **Great Expectations**: Dataset-level validation for batch data.

Suggested checks:

- Required fields present (no missing critical fields)
- Allowed ranges (e.g., heart rate, activity durations)
- Timestamp ordering and timezone consistency
- Outlier detection and basic anomaly thresholds

## Data Drift & Model Monitoring

Recommended tools:

- **Evidently AI**: Data drift, target drift, and performance reports
- **Arize**: Production monitoring and alerting

Suggested alerts:

- Distribution shift on key features
- Retrieval confidence drop over time
- Increased error rate or latency spikes

## Extension Points in This Repo

- **API layer**: Add validation to incoming payloads with Pydantic models
- **Retrieval layer**: Log retrieval scores and add drift checks on embeddings
- **Metrics endpoint**: Expose aggregated drift/quality stats

If you plan to add streaming sources (Kafka, Feast, TorchServe), add a separate `docker-compose.stream.yml` to keep the default demo lightweight.

