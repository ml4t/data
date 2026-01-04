# ML4T Data Production Deployment Guide

## Overview

This guide covers deploying ML4T Data in a production environment with proper CI/CD, monitoring, and security configurations.

## Prerequisites

- Python 3.9+
- Docker (optional but recommended)
- Redis/PostgreSQL for production storage (optional)
- Monitoring system (Prometheus/Grafana recommended)

## CI/CD Pipeline

### GitHub Actions Workflows

The project includes comprehensive CI/CD workflows:

#### 1. Continuous Integration (`ci.yml`)
- **Triggers**: Push to `main`/`develop`, Pull Requests
- **Jobs**:
  - **Test**: Runs on Python 3.9-3.12, includes coverage reporting
  - **Lint**: Code quality checks with ruff and mypy
  - **Security**: Bandit security scanning and dependency vulnerability checks
  - **Build**: Package building and validation
  - **Integration**: API and CLI testing

#### 2. Release Pipeline (`release.yml`)
- **Triggers**: GitHub releases
- **Jobs**:
  - Package building
  - PyPI publication (with trusted publishing)
  - GitHub release with signed artifacts

### Current CI Status
- **Coverage**: 75.57% (target: 75%+) ✅
- **Tests**: 479 passing, 1 failing (isolation issue)
- **Linting**: 220 warnings (non-blocking)
- **Security**: Clean (no high-severity issues)

## Production Configuration

### 1. Environment Variables

Create a `.env` file for production:

```bash
# Core settings
ML4T Data_LOG_LEVEL=INFO
ML4T Data_DATA_DIR=/opt/ml4t-data/data
ML4T Data_CACHE_DIR=/opt/ml4t-data/cache

# API settings
ML4T Data_API_HOST=0.0.0.0
ML4T Data_API_PORT=8000
ML4T Data_API_WORKERS=4
ML4T Data_API_KEY=your-secure-api-key-here

# Provider API keys
YAHOO_API_KEY=optional-yahoo-key
BINANCE_API_KEY=your-binance-key
BINANCE_API_SECRET=your-binance-secret
CRYPTOCOMPARE_API_KEY=your-cryptocompare-key

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
```

### 2. Production Config File

Use the provided production configuration:
```bash
cp examples/configs/production.yaml /opt/ml4t-data/config/production.yaml
```

### 3. Directory Structure

```
/opt/ml4t-data/
├── config/
│   └── production.yaml
├── data/
│   ├── equities/
│   ├── crypto/
│   └── cache/
├── logs/
│   └── ml4t-data.log
└── bin/
    └── ml4t-data
```

## Deployment Options

### Option 1: Direct Installation

```bash
# Install ML4T Data
pip install ml4t-data[api]

# Create directories
sudo mkdir -p /opt/ml4t-data/{data,config,logs}
sudo chown ml4t-data:ml4t-data /opt/ml4t-data -R

# Copy configuration
cp examples/configs/production.yaml /opt/ml4t-data/config/

# Start server
ml4t-data server --config /opt/ml4t-data/config/production.yaml
```

### Option 2: Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install -e .[api]

EXPOSE 8000 9090
CMD ["ml4t-data", "server", "--config", "/app/config/production.yaml"]
```

### Option 3: Systemd Service

```ini
[Unit]
Description=ML4T Data Data Manager API Server
After=network.target

[Service]
Type=exec
User=ml4t-data
Group=ml4t-data
WorkingDirectory=/opt/ml4t-data
Environment=PATH=/opt/ml4t-data/venv/bin
ExecStart=/opt/ml4t-data/venv/bin/ml4t-data server --config /opt/ml4t-data/config/production.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Monitoring Setup

### Prometheus Metrics

ML4T Data exposes metrics at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ml4t-data'
    static_configs:
      - targets: ['localhost:9090']
```

### Key Metrics to Monitor

- `ml4t-data_requests_total` - Total API requests
- `ml4t-data_request_duration_seconds` - Request latency
- `ml4t-data_data_fetch_duration_seconds` - Provider fetch times
- `ml4t-data_cache_hit_ratio` - Cache efficiency
- `ml4t-data_errors_total` - Error rates

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Detailed health with metrics
curl http://localhost:8000/health/detailed
```

## Security Considerations

### 1. API Security
- Use strong API keys (minimum 32 characters)
- Enable CORS restrictions for production
- Implement rate limiting
- Use HTTPS in production

### 2. Data Security
- Encrypt data at rest if required
- Secure API credentials in environment variables
- Regular security scans with `bandit`
- Dependency vulnerability monitoring

### 3. Network Security
- Firewall configuration
- VPN access for management
- Load balancer with SSL termination

## Performance Tuning

### 1. API Server
```yaml
# production.yaml
api:
  workers: 4  # 2x CPU cores
  timeout: 30
  max_concurrent_requests: 50
```

### 2. Caching
```yaml
performance:
  cache_enabled: true
  cache_ttl_seconds: 3600
  batch_size: 1000
```

### 3. Database Optimization
```yaml
storage:
  chunk_size_mb: 64
  compression: "gzip"
  batch_writes: true
```

## Backup and Recovery

### 1. Data Backup
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf /backup/ml4t-data-data-$DATE.tar.gz /opt/ml4t-data/data
find /backup -name "ml4t-data-data-*.tar.gz" -mtime +30 -delete
```

### 2. Configuration Backup
```bash
# Backup configuration
cp /opt/ml4t-data/config/* /backup/config/
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce batch sizes
   - Enable data compression
   - Implement data retention policies

2. **Slow API Responses**
   - Check provider rate limits
   - Enable caching
   - Review database queries

3. **Test Failures in CI**
   - Current known issue: 1 test isolation failure
   - Does not affect core functionality
   - Tracked for resolution in next sprint

### Logs and Debugging

```bash
# View logs
tail -f /var/log/ml4t-data/ml4t-data.log

# Debug mode (development only)
ML4T Data_LOG_LEVEL=DEBUG ml4t-data server
```

## Maintenance

### Regular Tasks

1. **Weekly**:
   - Review error logs
   - Check disk space
   - Update dependencies

2. **Monthly**:
   - Security scans
   - Performance review
   - Backup verification

3. **Quarterly**:
   - Dependency updates
   - Configuration review
   - Disaster recovery testing

## Next Steps

After deployment:

1. Set up monitoring dashboards
2. Configure alerting rules
3. Implement automated backups
4. Schedule regular maintenance
5. Plan for scaling based on usage

## Support and Documentation

- API Documentation: `/docs` endpoint
- Health Checks: `/health` endpoint
- Metrics: `/metrics` endpoint
- Issue Tracking: GitHub Issues
- Performance Monitoring: Prometheus/Grafana dashboards
