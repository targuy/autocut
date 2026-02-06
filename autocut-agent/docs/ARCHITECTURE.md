# AutoCut-Agent Architecture

## Overview

AutoCut-Agent is built using a layered architecture that promotes modularity, scalability, and maintainability. This document describes the system architecture, component interactions, and design decisions.

## Architecture Layers

### 1. Trigger Layer

The trigger layer is responsible for initiating task execution through various mechanisms:

#### Scheduler Trigger
- Uses APScheduler for cron-based and interval-based scheduling
- Supports multiple concurrent schedules
- Persistent job storage
- Timezone-aware scheduling

#### File Watcher Trigger
- Uses Watchdog library for file system monitoring
- Pattern matching for file types
- Debouncing to avoid duplicate triggers
- Recursive directory monitoring

#### API Trigger
- FastAPI REST endpoints for programmatic task submission
- Authentication and authorization
- Rate limiting
- Webhook support for external integrations

#### LLM Trigger
- Natural language command parsing using LangChain
- Intent recognition and parameter extraction
- Context-aware command execution
- Multi-turn conversation support

### 2. Agent Core / Orchestrator

The central coordination layer that manages all system components:

```python
class AgentOrchestrator:
    """Central orchestrator for the agent system."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.queue_manager = QueueManager(config.queues)
        self.resource_manager = ResourceManager(config.resources)
        self.trigger_manager = TriggerManager(config.triggers)
        self.executor_pool = ExecutorPool(config.workers)
        self.monitor = MonitoringSystem(config.monitoring)
    
    async def start(self):
        """Start all components."""
        await self.resource_manager.initialize()
        await self.queue_manager.start()
        await self.trigger_manager.start()
        await self.executor_pool.start()
        
    async def submit_task(self, task: Task):
        """Submit task for execution."""
        # Validate task
        # Add to appropriate queue
        # Notify resource manager
```

### 3. Queue Management Layer

Manages multiple queues with different execution strategies:

#### Queue Types

**FIFO Queue**
- First-in, first-out ordering
- Simple and predictable
- Good for sequential processing

**Priority Queue**
- Tasks have priority weights
- Higher priority tasks execute first
- Starvation prevention mechanisms

**Parallel Queue**
- Multiple tasks execute concurrently
- Respects resource constraints
- Load balancing across workers

#### Queue Manager

```python
class QueueManager:
    """Manages multiple task queues."""
    
    def __init__(self, configs: List[QueueConfig]):
        self.queues: Dict[str, Queue] = {}
        self.dispatcher = TaskDispatcher()
        
    async def add_task(self, queue_name: str, task: Task):
        """Add task to queue."""
        queue = self.queues[queue_name]
        await queue.push(task)
        await self.dispatcher.notify(queue_name)
        
    async def get_next_task(self, worker_id: str) -> Optional[Task]:
        """Get next available task for worker."""
        # Check worker capabilities
        # Find suitable queue
        # Check resource availability
        # Return task or None
```

### 4. Resource Management Layer

Handles exclusive and shared resource allocation:

#### Resource Types

**GPU Resources**
- Detection using CUDA/PyTorch
- Exclusive locking (one task at a time)
- VRAM monitoring
- Multi-GPU support

**CPU Resources**
- CPU core allocation
- Shared access (configurable concurrency)
- Load balancing

**Custom Resources**
- Plugin architecture for custom resource types
- License tokens, API quotas, etc.

#### Resource Manager

```python
class ResourceManager:
    """Manages resource allocation and locking."""
    
    def __init__(self, resources: List[ResourceConfig]):
        self.resources = self._initialize_resources(resources)
        self.lock_manager = RedisLockManager()  # or local locks
        
    async def acquire(self, resource_id: str, task_id: str) -> bool:
        """Acquire resource for task."""
        if await self.is_available(resource_id):
            await self.lock_manager.lock(resource_id, task_id)
            return True
        return False
        
    async def release(self, resource_id: str, task_id: str):
        """Release resource."""
        await self.lock_manager.unlock(resource_id, task_id)
```

### 5. Executor Layer

Executes Python programs in isolated environments:

#### Executor Components

**Runner**
- Subprocess management
- Output capture (stdout/stderr)
- Timeout enforcement
- Error handling

**Virtual Environment Manager**
- Venv activation
- Dependency isolation
- Environment variable injection

**Output Capture**
- Real-time log streaming
- Structured output parsing
- File output collection

```python
class ProgramExecutor:
    """Executes Python programs."""
    
    async def execute(self, task: Task) -> ExecutionResult:
        """Execute task program."""
        # Activate venv if needed
        # Prepare environment
        # Start subprocess
        # Capture output
        # Monitor resources
        # Handle completion/timeout
```

### 6. Monitoring Layer

Comprehensive monitoring and alerting:

#### Logging
- Structured JSON logging using structlog
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Context binding (task_id, queue_name, etc.)
- Log rotation and retention

#### Metrics
- Prometheus metrics export
- Task execution metrics (count, duration, success rate)
- Resource utilization metrics
- Queue depth and throughput

#### Alerting
- Multiple alert channels (email, webhook, Slack, Discord)
- Alert rules and conditions
- Alert throttling and deduplication
- Alert history tracking

```python
class MonitoringSystem:
    """Monitoring and alerting system."""
    
    def __init__(self, config: MonitoringConfig):
        self.logger = structlog.get_logger()
        self.metrics = PrometheusMetrics()
        self.alerter = AlertManager(config.alerts)
        
    async def log_event(self, event_type: str, **context):
        """Log structured event."""
        self.logger.info(event_type, **context)
        
    async def record_metric(self, metric_name: str, value: float, labels: Dict):
        """Record metric."""
        self.metrics.record(metric_name, value, labels)
```

### 7. API Layer

RESTful API for all operations:

#### FastAPI Application

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AutoCut-Agent API",
    description="Intelligent task orchestration system",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(queues_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")
app.include_router(status_router, prefix="/api/v1")
```

#### API Endpoints

**Queues**
- `POST /api/v1/queues` - Create queue
- `GET /api/v1/queues` - List queues
- `GET /api/v1/queues/{name}` - Get queue details
- `POST /api/v1/queues/{name}/pause` - Pause queue
- `POST /api/v1/queues/{name}/resume` - Resume queue
- `DELETE /api/v1/queues/{name}` - Delete queue

**Tasks**
- `POST /api/v1/tasks` - Submit task
- `GET /api/v1/tasks` - List tasks
- `GET /api/v1/tasks/{id}` - Get task details
- `GET /api/v1/tasks/{id}/logs` - Get task logs
- `GET /api/v1/tasks/{id}/output` - Get task output
- `DELETE /api/v1/tasks/{id}` - Cancel task

**Status**
- `GET /api/v1/status` - System status
- `GET /api/v1/health` - Health check
- `GET /api/v1/metrics` - Prometheus metrics

### 8. Web GUI Layer

React-based web interface (future implementation):

#### Components
- Dashboard overview
- Queue management interface
- Task monitor
- Configuration editor
- Log viewer
- Output browser with preview
- LLM chat interface

## Data Flow

### Task Submission Flow

```
1. Trigger detects condition (schedule, file, API call, etc.)
   ↓
2. Trigger creates Task object
   ↓
3. Task validated against program configuration
   ↓
4. Task added to appropriate Queue
   ↓
5. Queue notifies Dispatcher
   ↓
6. Dispatcher checks resource availability
   ↓
7. If resources available:
   - Resource Manager locks resources
   - Task assigned to Worker
   - Worker executes program
   - Output captured and stored
   - Resources released
   - Task marked complete
   ↓
8. Monitoring logs all events
   ↓
9. Alerts sent if configured
```

### Resource Locking Flow

```
1. Task requires GPU resource
   ↓
2. Dispatcher queries Resource Manager
   ↓
3. Resource Manager checks Redis lock
   ↓
4. If locked: Task waits in queue
   ↓
5. If available:
   - Acquire lock in Redis
   - Assign resource to task
   - Execute task
   - Release lock when complete
```

## Database Schema

### Tasks Table

```sql
CREATE TABLE tasks (
    id VARCHAR(36) PRIMARY KEY,
    queue_name VARCHAR(255) NOT NULL,
    program_path VARCHAR(1024) NOT NULL,
    status VARCHAR(20) NOT NULL,
    priority INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    
    config JSON NOT NULL,
    result JSON NULL,
    error TEXT NULL,
    
    INDEX idx_queue_status (queue_name, status),
    INDEX idx_status_created (status, created_at)
);
```

### Queues Table

```sql
CREATE TABLE queues (
    name VARCHAR(255) PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    workers INTEGER NOT NULL,
    priority INTEGER DEFAULT 0,
    status VARCHAR(20) NOT NULL,
    
    config JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Resources Table

```sql
CREATE TABLE resources (
    id VARCHAR(255) PRIMARY KEY,
    type VARCHAR(50) NOT NULL,
    exclusive BOOLEAN DEFAULT false,
    max_concurrent INTEGER DEFAULT 1,
    
    config JSON NOT NULL,
    status VARCHAR(20) NOT NULL,
    
    INDEX idx_type (type)
);
```

### Execution Logs Table

```sql
CREATE TABLE execution_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    task_id VARCHAR(36) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    context JSON NULL,
    
    INDEX idx_task_time (task_id, timestamp),
    INDEX idx_level (level),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);
```

## Configuration Management

### Configuration Structure

```yaml
# Hierarchical configuration with validation
agent:
  name: string
  workers: integer [1-32]
  log_level: enum [DEBUG, INFO, WARNING, ERROR, CRITICAL]

database:
  url: string (connection URL)
  pool_size: integer [1-100]

redis:
  url: string (connection URL)
  enabled: boolean

resources:
  - id: string
    type: enum [gpu, cpu, custom]
    exclusive: boolean
    max_concurrent: integer

queues:
  - name: string (unique)
    type: enum [fifo, priority, parallel]
    workers: integer
    resource_requirements: list

programs:
  - id: string (unique)
    path: string (file path)
    venv: string (optional)
    timeout: integer (seconds)

triggers:
  - type: enum [schedule, file_watcher, api, llm]
    enabled: boolean
    # Type-specific config

monitoring:
  metrics:
    enabled: boolean
    prometheus_port: integer
  logging:
    format: enum [json, text]
    level: enum
    file: string
  alerts:
    - type: enum [email, webhook, slack]
      # Alert-specific config

api:
  host: string (IP address)
  port: integer [1-65535]
  cors_enabled: boolean
  auth_enabled: boolean

llm:
  enabled: boolean
  provider: enum [openai, anthropic]
  model: string
```

### Configuration Loading

```python
from pydantic import BaseModel, Field, validator
import yaml

class AgentConfig(BaseModel):
    """Main configuration model."""
    
    agent: AgentSettings
    database: DatabaseSettings
    redis: RedisSettings
    resources: List[ResourceConfig]
    queues: List[QueueConfig]
    programs: List[ProgramConfig]
    triggers: List[TriggerConfig]
    monitoring: MonitoringConfig
    api: APIConfig
    llm: LLMConfig
    
    @validator("queues")
    def validate_queue_names_unique(cls, v):
        names = [q.name for q in v]
        if len(names) != len(set(names)):
            raise ValueError("Queue names must be unique")
        return v

def load_config(path: str) -> AgentConfig:
    """Load and validate configuration."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return AgentConfig(**data)
```

## Deployment Architecture

### Single Node Deployment

```
┌─────────────────────────────────────┐
│         Single Server               │
│  ┌──────────────────────────────┐  │
│  │    AutoCut-Agent Process     │  │
│  │  - Orchestrator              │  │
│  │  - Queue Manager             │  │
│  │  - Executor Pool             │  │
│  │  - API Server                │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │    SQLite Database           │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │    Optional: Redis           │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

### Distributed Deployment

```
┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │───▶│   API Nodes     │
│    (Nginx)      │    │  (Multiple)     │
└─────────────────┘    └────────┬────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼─────────┐    ┌────────▼────────┐    ┌───────▼─────────┐
│  Agent Node 1   │    │  Agent Node 2   │    │  Agent Node N   │
│  - Orchestrator │    │  - Orchestrator │    │  - Orchestrator │
│  - Queues       │    │  - Queues       │    │  - Queues       │
│  - Executors    │    │  - Executors    │    │  - Executors    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                       │
         └──────────────────────┼───────────────────────┘
                                │
         ┌──────────────────────┴───────────────────────┐
         │                                              │
  ┌──────▼─────────┐                      ┌────────────▼────────┐
  │   PostgreSQL   │                      │      Redis          │
  │   (Primary)    │◀────Replication────▶ │  (Cluster Mode)     │
  └────────────────┘                      └─────────────────────┘
```

## Security Considerations

### Authentication
- JWT token-based authentication
- API key support
- OAuth2 integration
- Role-based access control (RBAC)

### Authorization
- Queue-level permissions
- Task submission permissions
- Configuration edit permissions

### Data Security
- Encrypted connections (TLS/SSL)
- Secrets management (environment variables, vault)
- Database encryption at rest
- Audit logging

### Process Isolation
- Subprocess sandboxing
- Resource limits (CPU, memory, disk)
- Network isolation options
- Security scanning of executed programs

## Performance Optimization

### Caching Strategy
- Redis caching for frequently accessed data
- In-memory caching for configuration
- Queue state caching

### Database Optimization
- Appropriate indexes on frequently queried fields
- Connection pooling
- Query optimization
- Partitioning for large tables

### Resource Efficiency
- Worker pool sizing based on CPU cores
- Async I/O for all network operations
- Batch operations where possible
- Lazy loading of large objects

## Monitoring and Observability

### Metrics Collection
- Task execution metrics (count, duration, success rate)
- Queue depth and processing rate
- Resource utilization (CPU, memory, GPU)
- API request metrics (latency, error rate)

### Distributed Tracing
- OpenTelemetry integration
- Request ID propagation
- Span collection across components

### Logging Best Practices
- Structured JSON logs
- Correlation IDs for request tracking
- Log aggregation (ELK stack, Loki)
- Log sampling for high-volume events

## Future Enhancements

1. **Workflow DAGs**: Support for complex task dependencies
2. **Distributed Queues**: Kafka/RabbitMQ integration
3. **Plugin System**: Third-party extensions
4. **Advanced Scheduling**: Deadline-based scheduling
5. **Auto-scaling**: Kubernetes HPA integration
6. **Multi-tenancy**: Isolated environments per tenant
7. **Audit Trail**: Comprehensive audit logging
8. **Backup/Restore**: Configuration and state backup

## References

- [specifications-v1.md](../specifications-v1.md) - Complete specifications
- [AI-AGENTS-GUIDE.md](../AI-AGENTS-GUIDE.md) - AI assistant guide
- [README.md](../README.md) - Project overview
