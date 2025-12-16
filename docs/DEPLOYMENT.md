# 🚀 Deployment Guide

This guide covers multiple deployment options for the Multimodal RAG System.

## 📋 Table of Contents

1. [Streamlit Cloud (Frontend)](#streamlit-cloud-frontend)
2. [Docker Deployment](#docker-deployment)
3. [AWS EC2 Deployment](#aws-ec2-deployment)
4. [Troubleshooting](#troubleshooting)

---

## 🌐 Streamlit Cloud (Frontend)

Deploy the frontend to Streamlit Cloud for free, easy public access.

### Prerequisites

- GitHub account with your code pushed
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- Backend API deployed and accessible (optional for frontend-only demo)

### Step-by-Step Deployment

#### 1. Prepare Your Repository

Ensure these files exist in your repo:
- `frontend/requirements.txt` - Frontend-only dependencies (no ML libraries)
- `.python-version` - Specifies Python 3.11
- `runtime.txt` - Alternative Python version specification
- `.streamlit/config.toml` - Streamlit configuration

#### 2. Push to GitHub

```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

#### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Configure your app:
   - **Repository**: Your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `frontend/app.py`

4. Click "Advanced settings":
   - **Python version**: Select `3.11` (or leave empty to use `.python-version`)
   - **Requirements file**: Enter `frontend/requirements.txt`

5. Add Secrets (click "Secrets" section):
   ```toml
   API_URL = "https://your-backend-api-url.com"
   ```

   For local testing (API offline mode):
   ```toml
   API_URL = "http://localhost:8000"
   ```

6. Click "Deploy!"

#### 4. Wait for Deployment

- Initial deployment: 2-5 minutes
- Streamlit will install dependencies from `frontend/requirements.txt`
- Watch the deployment logs for any errors

### Configuration Files Explained

#### `frontend/requirements.txt`
```txt
# Minimal dependencies for frontend-only deployment
streamlit==1.28.2
requests==2.31.0
Pillow==10.1.0
pandas==2.1.3
python-dotenv==1.0.0
```

**Why separate from main requirements.txt?**
- Main `requirements.txt` includes heavy ML libraries (torch, transformers, CLIP)
- Frontend only needs web UI libraries
- Reduces deployment size from ~4GB to ~200MB
- Faster deployments and cold starts

#### `.python-version`
```
3.11
```

**Why Python 3.11?**
- Python 3.13 doesn't support torch 2.1.0
- Python 3.11 has the best compatibility with all dependencies
- Streamlit Cloud supports Python 3.11 natively

#### `.streamlit/config.toml`
```toml
[server]
headless = true
port = 8501

[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
```

Customizes Streamlit appearance and behavior.

### Common Issues and Solutions

#### Issue: "No matching distribution found for torch==2.1.0"

**Cause:** Using Python 3.13 or wrong requirements file

**Solution:**
1. Ensure `frontend/requirements.txt` is specified in Advanced settings
2. Verify Python version is 3.11
3. Check that `.python-version` contains `3.11`

#### Issue: "API Offline" error in the app

**Cause:** Backend API URL not configured or incorrect

**Solution:**
1. Go to app settings → Secrets
2. Add or update:
   ```toml
   API_URL = "https://your-backend-api-url.com"
   ```
3. Reboot the app

#### Issue: App keeps crashing during startup

**Cause:** Missing dependencies or wrong file path

**Solution:**
1. Check deployment logs for specific error
2. Verify `frontend/app.py` exists at specified path
3. Ensure all imports in `app.py` are in `frontend/requirements.txt`

### Auto-Deployment

Enable auto-deployment for automatic updates:

1. Go to app settings
2. Enable "Automatically update app when GitHub repository changes"
3. Every push to `main` branch will trigger redeployment

---

## 🐳 Docker Deployment

Deploy both frontend and backend together using Docker.

### Prerequisites

- Docker installed
- Docker Compose installed
- OpenAI API key

### Quick Start

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/multimodal-rag-system.git
   cd multimodal-rag-system
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Build and run**
   ```bash
   docker-compose up --build
   ```

4. **Access services**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Production Docker Deployment

For production, use specific tags and resource limits:

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  api:
    image: multimodal-rag-api:1.0.0
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  frontend:
    image: multimodal-rag-frontend:1.0.0
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G
```

---

## ☁️ AWS EC2 Deployment

Deploy the full stack on AWS EC2.

### Prerequisites

- AWS account
- EC2 instance (t2.medium or larger)
- Security group allowing ports 8000, 8501

### Deployment Steps

1. **Launch EC2 instance**
   - AMI: Amazon Linux 2 or Ubuntu 22.04
   - Instance type: t2.medium (minimum)
   - Storage: 20GB minimum

2. **SSH into instance**
   ```bash
   ssh -i your-key.pem ec2-user@your-instance-ip
   ```

3. **Install Docker**
   ```bash
   # Amazon Linux 2
   sudo yum update -y
   sudo yum install docker -y
   sudo service docker start
   sudo usermod -a -G docker ec2-user

   # Ubuntu
   sudo apt update
   sudo apt install docker.io docker-compose -y
   sudo systemctl start docker
   sudo usermod -a -G docker ubuntu
   ```

4. **Clone and configure**
   ```bash
   git clone https://github.com/yourusername/multimodal-rag-system.git
   cd multimodal-rag-system
   cp .env.example .env
   nano .env  # Add your API keys
   ```

5. **Deploy**
   ```bash
   docker-compose up -d
   ```

6. **Configure security group**
   - Inbound rules:
     - Port 8000: Backend API
     - Port 8501: Frontend
     - Port 22: SSH

### Monitoring and Logs

```bash
# View logs
docker-compose logs -f

# View specific service
docker-compose logs -f frontend

# Check status
docker-compose ps

# Restart services
docker-compose restart
```

---

## 🔧 Troubleshooting

### General Debugging

1. **Check logs first**
   ```bash
   # Streamlit Cloud: View logs in dashboard
   # Docker: docker-compose logs -f
   # Local: Check terminal output
   ```

2. **Verify environment variables**
   ```bash
   # Local/Docker
   cat .env

   # Streamlit Cloud
   Check app settings → Secrets
   ```

3. **Test API connectivity**
   ```bash
   curl http://your-api-url/health
   ```

### Dependency Issues

**Problem:** Package version conflicts

**Solution:**
```bash
# Create fresh virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # or frontend/requirements.txt
```

### Performance Issues

**Problem:** Slow response times

**Solutions:**
- Increase EC2 instance size
- Enable caching for embeddings
- Use GPU-enabled instances for ML workloads
- Implement request queuing

### Security

**Important:** Never commit `.env` files or API keys to GitHub!

```bash
# Verify .gitignore includes:
.env
.streamlit/secrets.toml
*.key
*.pem
```

---

## 📊 Deployment Comparison

| Option | Cost | Setup Time | Scalability | Best For |
|--------|------|------------|-------------|----------|
| **Streamlit Cloud** | Free | 5 min | Low | Frontend demos, prototypes |
| **Docker Local** | Free | 10 min | Low | Development, testing |
| **AWS EC2** | ~$20/month | 30 min | Medium | Small production apps |
| **AWS Lambda** | Pay-per-use | 1-2 hours | High | High-traffic APIs |

---

## 📝 Next Steps

1. **Monitor your deployment** - Set up CloudWatch (AWS) or check Streamlit logs
2. **Add custom domain** - Configure DNS for professional URLs
3. **Enable HTTPS** - Use Let's Encrypt or AWS Certificate Manager
4. **Set up CI/CD** - Automate deployments with GitHub Actions
5. **Add authentication** - Implement user login for sensitive data

---

## 🆘 Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/yourusername/multimodal-rag-system/issues)
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Docker Docs**: [docs.docker.com](https://docs.docker.com)

---

**Last Updated:** November 2024
