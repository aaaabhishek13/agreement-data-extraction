# Lease Agreement Extraction Tool - VM Deployment Guide (Direct Access)

## Overview
This guide will help you deploy the lease agreement extraction tool on a Virtual Machine (VM) for production use with direct access via IP and port4. **Monitor logs for suspicious activity**
   ```bash
   grep "ERROR\|CRITICAL" /opt/lease-extraction/logs/app.log
   ``` Nginx reverse proxy).

## Prerequisites
- Ubuntu 20.04/22.04 VM with at least:
  - 2 CPU cores
  - 4GB RAM
  - 20GB storage
  - Python 3.8+
- Root access to the VM
- API keys for OpenAI and/or Google Gemini

## Quick Deployment

### 1. Prepare Your VM
```bash
# SSH into your VM
ssh root@your-vm-ip

# Update the system
apt update && apt upgrade -y
```

### 2. Upload Application Code
Transfer your application files to the VM:
```bash
# Option 1: Using SCP
scp -r /path/to/doc-extraction root@your-vm-ip:/tmp/

# Option 2: Using Git (if you have a repository)
git clone your-repo-url /tmp/doc-extraction
```

### 3. Move Code to Application Directory
```bash
mkdir -p /opt/lease-extraction
cp -r /tmp/doc-extraction/* /opt/lease-extraction/
cd /opt/lease-extraction
```

### 4. Configure Environment Variables
```bash
# Copy and edit the production environment file
cp deployment/.env.production .env
nano .env

# Update the following:
# - OPENAI_API_KEY=your_actual_openai_key
# - GEMINI_API_KEY=your_actual_gemini_key
# - SECRET_KEY=your_long_random_secret_key
```

### 5. Run Deployment Script
```bash
chmod +x deployment/deploy.sh
./deployment/deploy.sh
```

### 6. Verify Deployment
```bash
# Check application status
supervisorctl status lease-extraction

# Test the application
curl http://localhost:5001/api/test

# Test from external machine
curl http://your-vm-ip:5001/api/test
```

## Manual Step-by-Step Deployment

If you prefer manual deployment or need to troubleshoot:

### 1. Install System Dependencies
```bash
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv supervisor git curl
```

### 2. Create Application User
```bash
useradd -m -s /bin/bash leaseapp
```

### 3. Set Up Application Directory
```bash
mkdir -p /opt/lease-extraction
chown leaseapp:leaseapp /opt/lease-extraction
# Copy your application files here
```

### 4. Create Virtual Environment
```bash
sudo -u leaseapp python3 -m venv /opt/lease-extraction/venv
sudo -u leaseapp /opt/lease-extraction/venv/bin/pip install -r deployment/requirements-prod.txt
```

### 5. Configure Services
```bash
# Supervisor configuration
cp deployment/supervisor.conf /etc/supervisor/conf.d/lease-extraction.conf
systemctl restart supervisor
```

### 6. Start Application
```bash
supervisorctl reread
supervisorctl update
supervisorctl start lease-extraction
```

## Post-Deployment Configuration

### Firewall Configuration
```bash
ufw allow ssh
ufw allow 5001
ufw enable
```

### Monitoring Setup
```bash
# Make health check script executable
chmod +x deployment/health-check.sh

# Add to crontab for monitoring
crontab -e
# Add: */5 * * * * /opt/lease-extraction/deployment/health-check.sh
```

## Maintenance Commands

### View Application Logs
```bash
# Application logs
tail -f /opt/lease-extraction/logs/app.log

# Extraction logs
tail -f /opt/lease-extraction/extraction.log
```

### Restart Services
```bash
# Restart application
supervisorctl restart lease-extraction

# Restart supervisor
systemctl restart supervisor
```

### Update Application
```bash
# Stop application
supervisorctl stop lease-extraction

# Update code (git pull or copy new files)
# Install new dependencies if needed
sudo -u leaseapp /opt/lease-extraction/venv/bin/pip install -r deployment/requirements-prod.txt

# Start application
supervisorctl start lease-extraction
```

## Troubleshooting

### Common Issues

1. **Application won't start**
   ```bash
   # Check supervisor logs
   supervisorctl tail -f lease-extraction
   
   # Check if port is in use
   netstat -tlnp | grep :5000
   ```

2. **502 Bad Gateway / Connection Refused**
   ```bash
   # Check if application is running
   supervisorctl status lease-extraction
   
   # Check if port is accessible
   netstat -tlnp | grep :5001
   
   # Test locally
   curl http://localhost:5001/api/test
   ```

3. **Permission Issues**
   ```bash
   # Fix permissions
   chown -R leaseapp:leaseapp /opt/lease-extraction
   chmod -R 755 /opt/lease-extraction
   chmod -R 777 /opt/lease-extraction/uploads
   ```

4. **Out of Memory**
   ```bash
   # Check memory usage
   free -h
   
   # Reduce Gunicorn workers in supervisor.conf
   nano /etc/supervisor/conf.d/lease-extraction.conf
   ```

### Performance Tuning

1. **For High Traffic**
   - Increase Gunicorn workers in `supervisor.conf`
   - Adjust nginx worker processes
   - Consider adding Redis for session storage

2. **For Large Files**
   - Increase timeout values in supervisor.conf
   - Monitor disk space in uploads directory

## Security Considerations

1. **Keep system updated**
   ```bash
   apt update && apt upgrade -y
   ```

2. **Regular security audits**
   ```bash
   # Check for security updates
   unattended-upgrades --dry-run
   ```

3. **Monitor logs for suspicious activity**
   ```bash
   grep "404\|403\|500" /var/log/nginx/access.log
   ```

4. **Use strong passwords and keys**
   - Generate strong SECRET_KEY
   - Secure API keys
   - Use SSH key authentication

## Backup Strategy

```bash
# Create backup script
cat > /opt/lease-extraction/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/lease-extraction"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/app_$DATE.tar.gz /opt/lease-extraction
find $BACKUP_DIR -name "app_*.tar.gz" -mtime +7 -delete
EOF

chmod +x /opt/lease-extraction/backup.sh

# Add to crontab for daily backups
echo "0 2 * * * /opt/lease-extraction/backup.sh" | crontab -
```

## Scaling Options

1. **Horizontal Scaling**: Deploy multiple instances behind a load balancer
2. **Database**: Add PostgreSQL/MySQL for storing results
3. **Queue System**: Add Celery + Redis for background processing
4. **CDN**: Use CloudFlare or AWS CloudFront for static assets

Your application will be available at:
- HTTP: `http://your-vm-ip:5001`
