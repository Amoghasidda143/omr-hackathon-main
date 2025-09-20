#!/bin/bash
# OMR Evaluation System - Docker Entrypoint Script

set -e

echo "🚀 Starting OMR Evaluation System..."

# Wait for database to be ready (if using external DB)
echo "⏳ Waiting for database..."
sleep 5

# Create database tables
echo "📊 Setting up database..."
python -c "
try:
    from models.database import create_tables
    create_tables()
    print('✅ Database tables created successfully')
except Exception as e:
    print(f'⚠️ Database setup warning: {e}')
"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads results logs models answer_keys static

# Set permissions
chmod -R 755 uploads results logs models answer_keys static

# Check if we're running backend or frontend
if [ "$1" = "uvicorn" ]; then
    echo "🚀 Starting FastAPI backend..."
    exec "$@"
elif [ "$1" = "streamlit" ]; then
    echo "🎨 Starting Streamlit frontend..."
    exec "$@"
else
    echo "❌ Unknown command: $1"
    echo "Available commands: uvicorn, streamlit"
    exit 1
fi
