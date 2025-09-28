#!/bin/bash

# VM Deployment Script for Lease Extraction Tool
# Run this script as root on a fresh Ubuntu VM

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting deployment of Lease Extraction Tool${NC}"

# Update system packages
echo -e "${YELLOW}📦 Updating system packages...${NC}"
apt update && apt upgrade -y

# Install system dependencies
echo -e "${YELLOW}📦 Installing system dependencies...${NC}"
apt install -y python3 python3-pip python3-venv supervisor git curl

# Create application user
echo -e "${YELLOW}👤 Creating application user...${NC}"
if ! id "leaseapp" &>/dev/null; then
    useradd -m -s /bin/bash leaseapp
fi

# Create application directory
echo -e "${YELLOW}📁 Setting up application directory...${NC}"
APP_DIR="/home/ubuntu/agreement-data-extraction"
mkdir -p $APP_DIR
chown leaseapp:leaseapp $APP_DIR

# Clone or copy application code (adjust as needed)
echo -e "${YELLOW}📥 Setting up application code...${NC}"
# If using git:
# git clone <your-repo-url> $APP_DIR
# For now, assume code is copied manually to $APP_DIR

# Create virtual environment
echo -e "${YELLOW}🐍 Creating Python virtual environment...${NC}"
sudo -u leaseapp python3 -m venv $APP_DIR/venv

# Install Python dependencies
echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
sudo -u leaseapp $APP_DIR/venv/bin/pip install --upgrade pip
sudo -u leaseapp $APP_DIR/venv/bin/pip install -r $APP_DIR/deployment/requirements-prod.txt

# Create necessary directories
echo -e "${YELLOW}📁 Creating application directories...${NC}"
sudo -u leaseapp mkdir -p $APP_DIR/uploads
sudo -u leaseapp mkdir -p $APP_DIR/logs
sudo -u leaseapp mkdir -p $APP_DIR/static

# Set up environment variables
echo -e "${YELLOW}⚙️ Setting up environment configuration...${NC}"
sudo -u leaseapp cp $APP_DIR/deployment/.env.production $APP_DIR/.env

# Configure Supervisor
echo -e "${YELLOW}⚙️ Configuring Supervisor...${NC}"
cp $APP_DIR/deployment/supervisor.conf /etc/supervisor/conf.d/lease-extraction.conf
systemctl enable supervisor
systemctl restart supervisor

# Start services
echo -e "${YELLOW}🚀 Starting services...${NC}"
supervisorctl reread
supervisorctl update
supervisorctl start lease-extraction

# Setup SSL (optional - uncomment if you have a domain)
# echo -e "${YELLOW}🔒 Setting up SSL with Certbot...${NC}"
# apt install -y certbot python3-certbot-nginx
# certbot --nginx -d your-domain.com

# Setup log rotation
echo -e "${YELLOW}📝 Setting up log rotation...${NC}"
cp $APP_DIR/deployment/logrotate.conf /etc/logrotate.d/lease-extraction

# Set up firewall
echo -e "${YELLOW}🔥 Configuring firewall...${NC}"
ufw allow ssh
ufw allow 5001
ufw --force enable

# Final permissions
chown -R leaseapp:leaseapp $APP_DIR
chmod -R 755 $APP_DIR
chmod -R 777 $APP_DIR/uploads
chmod -R 755 $APP_DIR/logs

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${GREEN}🌐 Your application should be available at: http://your-vm-ip:5001${NC}"
echo -e "${GREEN}📊 Check status with: supervisorctl status lease-extraction${NC}"
echo -e "${GREEN}📝 View logs with: tail -f $APP_DIR/logs/app.log${NC}"
