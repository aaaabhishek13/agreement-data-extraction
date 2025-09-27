#!/bin/bash

# Health check script for monitoring
# Can be used with monitoring systems like Nagios, Zabbix, etc.

APP_URL="http://localhost:5001/api/test"
TIMEOUT=10

response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT $APP_URL)

if [ "$response" == "200" ]; then
    echo "OK: Application is responding"
    exit 0
else
    echo "CRITICAL: Application not responding (HTTP $response)"
    exit 2
fi
