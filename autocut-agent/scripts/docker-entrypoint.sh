#!/bin/bash
# Docker entrypoint script for AutoCut-Agent

set -e

echo "Starting AutoCut-Agent..."

# Wait for PostgreSQL to be ready (if using PostgreSQL)
if [ -n "$AGENT_DATABASE_URL" ] && [[ "$AGENT_DATABASE_URL" == postgresql* ]]; then
    echo "Waiting for PostgreSQL..."
    until pg_isready -h "${AGENT_DATABASE_URL##*@}" -p 5432 &> /dev/null; do
        echo "PostgreSQL is unavailable - sleeping"
        sleep 2
    done
    echo "PostgreSQL is up"
fi

# Wait for Redis to be ready (if using Redis)
if [ -n "$AGENT_REDIS_URL" ] && [ "$AGENT_REDIS_ENABLED" = "true" ]; then
    echo "Waiting for Redis..."
    redis_host=$(echo "$AGENT_REDIS_URL" | sed -n 's/.*:\/\/\([^:]*\).*/\1/p')
    until redis-cli -h "$redis_host" ping &> /dev/null; do
        echo "Redis is unavailable - sleeping"
        sleep 2
    done
    echo "Redis is up"
fi

# Run database migrations (if needed)
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    cd /app/src && alembic upgrade head
    echo "Migrations complete"
fi

# Execute the command passed to docker run
exec python -m agent.cli "$@"
