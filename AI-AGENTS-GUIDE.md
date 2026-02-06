# AI Agents Guide for AutoCut-Agent Development

This guide provides comprehensive instructions for AI coding assistants (Claude, ChatGPT, Gemini, GitHub Copilot, Cursor, Aider, etc.) to accelerate the development of the AutoCut-Agent project.

## Table of Contents

1. [Quick Start for AI Agents](#quick-start-for-ai-agents)
2. [Project Context](#project-context)
3. [Development Guidelines](#development-guidelines)
4. [Code Generation Patterns](#code-generation-patterns)
5. [Testing Strategies](#testing-strategies)
6. [Common Tasks](#common-tasks)
7. [Architecture Decisions](#architecture-decisions)
8. [AI-Specific Tips](#ai-specific-tips)

## Quick Start for AI Agents

### Initial Context Loading

When starting work on AutoCut-Agent, load these files first:

```
1. specifications-v1.md      # Complete project specifications
2. README.md                  # Project overview and setup
3. ARCHITECTURE.md            # System architecture (when created)
4. pyproject.toml             # Dependencies and project config
5. src/agent/config.py        # Configuration structure
```

### Project Goal Summary

AutoCut-Agent is a **Python-based task orchestration system** that:
- Executes Python programs via multiple triggers (schedule, events, API, GUI, LLM)
- Manages parallel queues with resource locking (especially GPU/CUDA)
- Provides web GUI for administration and monitoring
- Supports LLM-based natural language control

### Technology Stack at a Glance

```python
# Core
python >= 3.10
fastapi >= 0.104      # Web framework
sqlalchemy >= 2.0     # ORM
pydantic >= 2.0       # Validation

# Task Management
apscheduler >= 3.10   # Scheduling
watchdog >= 3.0       # File monitoring
celery >= 5.3         # Optional distributed tasks

# LLM
langchain >= 0.1      # LLM framework
openai                # OpenAI API
anthropic             # Claude API

# Database
sqlite3 (built-in)    # Default storage
redis >= 7.0          # Caching & locking

# Frontend
react >= 18           # UI framework
tailwindcss           # Styling
axios                 # HTTP client

# Testing
pytest >= 7.4
pytest-asyncio
httpx                 # Async HTTP testing
```

## Project Context

### Directory Structure

```
autocut-agent/
├── .github/
│   └── copilot-instructions.md    # GitHub Copilot specific instructions
├── .vscode/
│   ├── settings.json              # VSCode workspace settings
│   ├── launch.json                # Debug configurations
│   └── tasks.json                 # Common tasks
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── orchestrator.py    # Main agent orchestrator
│       │   ├── config.py          # Configuration management
│       │   └── state.py           # State persistence
│       ├── triggers/
│       │   ├── __init__.py
│       │   ├── scheduler.py       # Cron/interval scheduling
│       │   ├── watcher.py         # File system monitoring
│       │   ├── api.py             # REST API triggers
│       │   └── llm.py             # LLM-based triggers
│       ├── queue/
│       │   ├── __init__.py
│       │   ├── manager.py         # Queue management
│       │   ├── worker.py          # Task workers
│       │   └── models.py          # Queue data models
│       ├── resources/
│       │   ├── __init__.py
│       │   ├── manager.py         # Resource management
│       │   ├── gpu.py             # GPU detection/locking
│       │   └── locks.py           # Distributed locking
│       ├── executor/
│       │   ├── __init__.py
│       │   ├── runner.py          # Program execution
│       │   ├── venv.py            # Virtual environment management
│       │   └── capture.py         # Output capture
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── logger.py          # Structured logging
│       │   ├── metrics.py         # Metrics collection
│       │   └── alerts.py          # Alert management
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py            # FastAPI app
│       │   ├── routes/            # API endpoints
│       │   │   ├── queues.py
│       │   │   ├── tasks.py
│       │   │   ├── config.py
│       │   │   └── status.py
│       │   ├── websocket.py       # WebSocket for real-time
│       │   └── auth.py            # Authentication
│       └── gui/
│           ├── __init__.py
│           └── app/               # React frontend
│               ├── public/
│               ├── src/
│               │   ├── components/
│               │   ├── pages/
│               │   ├── api/
│               │   └── App.jsx
│               └── package.json
├── configs/
│   ├── default.yaml               # Default configuration
│   ├── development.yaml           # Dev environment
│   ├── production.yaml            # Prod environment
│   └── examples/                  # Example configs
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── api.md                     # API documentation
│   ├── configuration.md           # Config guide
│   ├── deployment.md              # Deployment guide
│   └── examples/                  # Usage examples
├── examples/
│   ├── simple_task.py             # Basic task example
│   ├── gpu_task.py                # GPU-locked task
│   └── mcp_tool.py                # MCP tool example
├── scripts/
│   ├── setup.sh                   # Linux/macOS setup
│   ├── setup.ps1                  # Windows setup
│   └── docker-entrypoint.sh       # Container entrypoint
├── .cursorrules                   # Cursor AI instructions
├── .aider.conf.yml                # Aider configuration
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                 # Poetry config
├── requirements.txt               # Pip fallback
├── README.md
├── ARCHITECTURE.md
├── CONTRIBUTING.md
└── LICENSE
```

### Key Architectural Principles

1. **Modularity**: Each component (triggers, queues, resources) is independent
2. **Async-First**: Use asyncio throughout for concurrency
3. **Type Safety**: Use Pydantic models and type hints everywhere
4. **Configuration**: YAML-based with Pydantic validation
5. **Testing**: High coverage with pytest, mock external dependencies
6. **Logging**: Structured JSON logging with context
7. **API-First**: All operations available via REST API
8. **Plugin-Ready**: Design for extensibility

## Development Guidelines

### Python Style Guide

```python
# Use type hints everywhere
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

async def process_task(
    task_id: str,
    queue_name: str,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Process a task from the queue.
    
    Args:
        task_id: Unique task identifier
        queue_name: Name of the queue containing the task
        timeout: Optional execution timeout in seconds
        
    Returns:
        Dictionary containing task result and metadata
        
    Raises:
        TaskNotFoundError: If task_id doesn't exist
        ExecutionError: If task execution fails
    """
    pass

# Use Pydantic for data validation
class TaskConfig(BaseModel):
    """Configuration for a task execution."""
    
    program_path: str
    venv_path: Optional[str] = None
    timeout: int = 3600
    retry_attempts: int = 3
    
    class Config:
        frozen = True  # Immutable
```

### Error Handling Pattern

```python
from typing import Union
import structlog

logger = structlog.get_logger(__name__)

class AgentException(Exception):
    """Base exception for all agent errors."""
    pass

class ResourceLockError(AgentException):
    """Raised when resource cannot be acquired."""
    pass

async def acquire_gpu(gpu_id: str, timeout: int = 30) -> Union[bool, None]:
    """Acquire exclusive GPU lock with timeout."""
    try:
        lock = await resource_manager.lock(
            resource_type="gpu",
            resource_id=gpu_id,
            timeout=timeout
        )
        logger.info("gpu_acquired", gpu_id=gpu_id)
        return True
    except ResourceLockError as e:
        logger.error("gpu_lock_failed", gpu_id=gpu_id, error=str(e))
        raise
    except Exception as e:
        logger.exception("unexpected_error", gpu_id=gpu_id)
        raise AgentException(f"Failed to acquire GPU {gpu_id}") from e
```

### Configuration Pattern

```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
import yaml

class GPUConfig(BaseModel):
    """GPU resource configuration."""
    id: str
    exclusive: bool = True
    max_concurrent: int = 1

class AgentConfig(BaseSettings):
    """Main agent configuration."""
    
    # Agent settings
    name: str = "AutoCut Agent"
    workers: int = Field(default=4, ge=1, le=32)
    log_level: str = Field(default="INFO")
    
    # Database
    database_url: str = "sqlite:///agent.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # GPU resources
    gpus: List[GPUConfig] = []
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_secret_key: str = Field(..., env="API_SECRET_KEY")
    
    class Config:
        env_file = ".env"
        env_prefix = "AGENT_"
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()

def load_config(path: str = "config.yaml") -> AgentConfig:
    """Load and validate configuration from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return AgentConfig(**data)
```

### Database Models Pattern

```python
from sqlalchemy import Column, String, Integer, DateTime, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum

Base = declarative_base()

class TaskStatus(str, enum.Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Task(Base):
    """Task database model."""
    
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True)
    queue_name = Column(String, nullable=False, index=True)
    program_path = Column(String, nullable=False)
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, index=True)
    
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    config = Column(JSON, nullable=False)
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "queue_name": self.queue_name,
            "program_path": self.program_path,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config": self.config,
            "result": self.result,
            "error": self.error,
        }
```

### API Endpoint Pattern

```python
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/api/v1/queues", tags=["queues"])

class QueueCreateRequest(BaseModel):
    """Request model for creating a queue."""
    name: str
    workers: int = 1
    priority: int = 0

class QueueResponse(BaseModel):
    """Response model for queue information."""
    name: str
    workers: int
    priority: int
    pending_count: int
    running_count: int
    status: str

@router.post("/", response_model=QueueResponse, status_code=status.HTTP_201_CREATED)
async def create_queue(
    request: QueueCreateRequest,
    queue_manager = Depends(get_queue_manager)
) -> QueueResponse:
    """Create a new task queue.
    
    Args:
        request: Queue creation parameters
        queue_manager: Injected queue manager dependency
        
    Returns:
        Created queue information
        
    Raises:
        409: Queue with same name already exists
    """
    try:
        queue = await queue_manager.create_queue(
            name=request.name,
            workers=request.workers,
            priority=request.priority
        )
        return QueueResponse(
            name=queue.name,
            workers=queue.workers,
            priority=queue.priority,
            pending_count=0,
            running_count=0,
            status="active"
        )
    except QueueExistsError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Queue '{request.name}' already exists"
        )

@router.get("/", response_model=List[QueueResponse])
async def list_queues(
    queue_manager = Depends(get_queue_manager)
) -> List[QueueResponse]:
    """List all queues with their current status."""
    queues = await queue_manager.list_queues()
    return [
        QueueResponse(
            name=q.name,
            workers=q.workers,
            priority=q.priority,
            pending_count=await q.pending_count(),
            running_count=await q.running_count(),
            status=q.status
        )
        for q in queues
    ]
```

### Testing Pattern

```python
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
async def mock_queue_manager():
    """Mock queue manager for testing."""
    manager = AsyncMock()
    manager.create_queue.return_value = MagicMock(
        name="test_queue",
        workers=2,
        priority=1,
        status="active"
    )
    return manager

@pytest.fixture
async def app(mock_queue_manager):
    """FastAPI test application."""
    from agent.api.main import create_app
    app = create_app()
    app.dependency_overrides[get_queue_manager] = lambda: mock_queue_manager
    return app

@pytest.mark.asyncio
async def test_create_queue(app, mock_queue_manager):
    """Test queue creation endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/queues/",
            json={
                "name": "test_queue",
                "workers": 2,
                "priority": 1
            }
        )
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test_queue"
    assert data["workers"] == 2
    
    mock_queue_manager.create_queue.assert_called_once_with(
        name="test_queue",
        workers=2,
        priority=1
    )

@pytest.mark.asyncio
async def test_create_duplicate_queue(app, mock_queue_manager):
    """Test creating queue with duplicate name."""
    mock_queue_manager.create_queue.side_effect = QueueExistsError("Already exists")
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/queues/",
            json={"name": "duplicate_queue", "workers": 1}
        )
    
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()
```

## Code Generation Patterns

### When Creating a New Module

```python
"""
Module: agent.queue.manager
Description: Manages task queues with priority and resource constraints.

This module provides the QueueManager class responsible for:
- Creating and managing multiple task queues
- Dispatching tasks to appropriate workers
- Handling queue lifecycle (pause, resume, delete)
- Coordinating with resource manager for locks
"""

from typing import Dict, List, Optional
import asyncio
import structlog
from pydantic import BaseModel

from agent.core.config import QueueConfig
from agent.queue.models import Task, TaskStatus
from agent.resources.manager import ResourceManager

logger = structlog.get_logger(__name__)

class QueueManager:
    """Manages multiple task queues with coordinated execution."""
    
    def __init__(
        self,
        resource_manager: ResourceManager,
        config: QueueConfig
    ):
        """Initialize queue manager.
        
        Args:
            resource_manager: Resource management instance
            config: Queue configuration
        """
        self.resource_manager = resource_manager
        self.config = config
        self.queues: Dict[str, Queue] = {}
        self._lock = asyncio.Lock()
        
        logger.info("queue_manager_initialized", config=config)
    
    async def create_queue(
        self,
        name: str,
        workers: int = 1,
        priority: int = 0
    ) -> "Queue":
        """Create a new task queue.
        
        Args:
            name: Unique queue name
            workers: Number of concurrent workers
            priority: Queue priority (higher = more important)
            
        Returns:
            Created Queue instance
            
        Raises:
            QueueExistsError: If queue name already exists
        """
        async with self._lock:
            if name in self.queues:
                raise QueueExistsError(f"Queue '{name}' already exists")
            
            queue = Queue(
                name=name,
                workers=workers,
                priority=priority,
                resource_manager=self.resource_manager
            )
            self.queues[name] = queue
            
            logger.info("queue_created", name=name, workers=workers, priority=priority)
            return queue
    
    # ... more methods ...
```

### When Adding an API Endpoint

Follow this checklist:

1. **Define Pydantic models** for request and response
2. **Create route function** with proper type hints
3. **Add docstring** with Args, Returns, Raises
4. **Implement business logic** with error handling
5. **Add logging** for debugging
6. **Write tests** covering success and error cases
7. **Update OpenAPI** tags and descriptions

### When Adding Configuration

1. **Add to Pydantic model** in `config.py`
2. **Add validator** if needed
3. **Update default.yaml** with new field and comment
4. **Update docs** in `configuration.md`
5. **Add test** for configuration validation

## Testing Strategies

### Unit Tests

```python
# Test individual functions/methods in isolation
# Mock all external dependencies

@pytest.mark.asyncio
async def test_task_execution():
    """Test task execution in isolation."""
    mock_executor = AsyncMock()
    mock_executor.run.return_value = {"status": "success"}
    
    task = Task(id="test-1", program_path="/test.py")
    result = await execute_task(task, mock_executor)
    
    assert result["status"] == "success"
    mock_executor.run.assert_called_once()
```

### Integration Tests

```python
# Test multiple components working together
# Use real database (SQLite in memory)

@pytest.mark.asyncio
async def test_queue_to_executor_flow(db_session):
    """Test task flowing from queue to executor."""
    queue_manager = QueueManager(db_session)
    executor = Executor()
    
    # Create queue and add task
    queue = await queue_manager.create_queue("test")
    task = await queue.add_task("/test.py")
    
    # Execute task
    result = await executor.process_next(queue)
    
    # Verify task completed
    updated_task = await db_session.get(Task, task.id)
    assert updated_task.status == TaskStatus.COMPLETED
```

### E2E Tests

```python
# Test complete flows through API
# Use test server

@pytest.mark.asyncio
async def test_api_task_submission_and_execution(test_client):
    """Test submitting and executing task via API."""
    # Create queue
    response = await test_client.post(
        "/api/v1/queues/",
        json={"name": "test_queue", "workers": 1}
    )
    assert response.status_code == 201
    
    # Submit task
    response = await test_client.post(
        "/api/v1/tasks/",
        json={
            "queue_name": "test_queue",
            "program_path": "/test.py"
        }
    )
    task_id = response.json()["id"]
    
    # Wait for completion
    await asyncio.sleep(2)
    
    # Check status
    response = await test_client.get(f"/api/v1/tasks/{task_id}")
    assert response.json()["status"] == "completed"
```

## Common Tasks

### Task 1: Add a New Queue Type

```python
# 1. Create new queue class inheriting from BaseQueue
class PriorityQueue(BaseQueue):
    """Queue with priority-based task selection."""
    
    async def get_next_task(self) -> Optional[Task]:
        """Get highest priority pending task."""
        # Implementation
        pass

# 2. Register in queue factory
QUEUE_TYPES = {
    "fifo": FIFOQueue,
    "priority": PriorityQueue,  # Add here
    "parallel": ParallelQueue,
}

# 3. Add to configuration schema
class QueueConfig(BaseModel):
    type: Literal["fifo", "priority", "parallel"]  # Add here

# 4. Write tests
@pytest.mark.asyncio
async def test_priority_queue_ordering():
    """Test priority queue returns highest priority first."""
    # Test implementation
    pass

# 5. Update documentation
```

### Task 2: Add a New Trigger Type

```python
# 1. Create trigger class
class HTTPWebhookTrigger(BaseTrigger):
    """Trigger tasks via HTTP webhook."""
    
    def __init__(self, config: WebhookConfig):
        self.config = config
        self.app = FastAPI()
        
    async def start(self):
        """Start webhook server."""
        pass
    
    async def stop(self):
        """Stop webhook server."""
        pass

# 2. Register trigger
TRIGGER_TYPES = {
    "schedule": ScheduleTrigger,
    "file_watcher": FileWatcherTrigger,
    "webhook": HTTPWebhookTrigger,  # Add here
}

# 3. Add configuration
class WebhookTriggerConfig(BaseModel):
    type: Literal["webhook"]
    port: int = 9000
    secret: str

# 4. Test
@pytest.mark.asyncio
async def test_webhook_trigger(test_client):
    """Test webhook triggers task."""
    # Implementation
    pass
```

### Task 3: Add Resource Type

```python
# 1. Create resource detector
class CPUResourceDetector:
    """Detect available CPU cores."""
    
    @staticmethod
    def detect() -> List[Resource]:
        import psutil
        return [
            Resource(
                id="cpu",
                type="cpu",
                capacity=psutil.cpu_count()
            )
        ]

# 2. Register detector
RESOURCE_DETECTORS = {
    "gpu": GPUDetector,
    "cpu": CPUResourceDetector,  # Add here
}

# 3. Add locking strategy
class CPULockStrategy(BaseLockStrategy):
    """CPU core allocation strategy."""
    
    async def acquire(self, count: int) -> List[str]:
        """Acquire N CPU cores."""
        pass

# 4. Test
@pytest.mark.asyncio
async def test_cpu_resource_allocation():
    """Test CPU core allocation."""
    # Implementation
    pass
```

## Architecture Decisions

### Why FastAPI?

- **Performance**: Asynchronous by default, very fast
- **Developer Experience**: Auto OpenAPI docs, type validation
- **Modern**: Python 3.10+ features, async/await native
- **Community**: Very active, 70k+ stars

### Why SQLAlchemy?

- **ORM**: Clean Python objects, no SQL
- **Database Agnostic**: SQLite, PostgreSQL, MySQL, etc.
- **Async Support**: Full async/await in 2.0+
- **Migrations**: Alembic integration

### Why Pydantic?

- **Validation**: Automatic data validation
- **Type Safety**: Runtime type checking
- **Configuration**: Clean settings management
- **Serialization**: JSON, dict conversion

### Why Redis?

- **Locking**: Distributed locks for resource coordination
- **Caching**: Fast access to frequently used data
- **Pub/Sub**: Real-time event broadcasting
- **Simple**: Easy to setup and use

### Why LangChain?

- **LLM Abstraction**: Works with multiple LLM providers
- **Agents**: Built-in agent framework
- **Memory**: Conversation history management
- **Community**: Very active, lots of examples

## AI-Specific Tips

### For Claude (Anthropic)

Claude excels at:
- **Long-form planning**: Ask for detailed implementation plans
- **Architecture design**: Request system design with tradeoffs
- **Code review**: Paste code and ask for improvements
- **Documentation**: Generate comprehensive docs

Example prompts:
```
"Design the QueueManager class with full implementation, including error handling, logging, and tests"

"Review this code for potential race conditions and suggest improvements"

"Write comprehensive API documentation for the /queues endpoints"
```

### For ChatGPT (OpenAI)

ChatGPT excels at:
- **Quick prototypes**: Fast code generation
- **Debugging**: Explain error messages
- **Refactoring**: Improve existing code
- **Examples**: Generate usage examples

Example prompts:
```
"Write a quick prototype of a file watcher that monitors a directory"

"This error occurs: [paste error]. What's causing it and how do I fix it?"

"Refactor this function to be more Pythonic and add type hints"
```

### For Gemini (Google)

Gemini excels at:
- **Multimodal**: Can analyze images/diagrams
- **Research**: Finding libraries and best practices
- **Data processing**: Working with structured data
- **Testing**: Generating test cases

Example prompts:
```
"What's the best Python library for distributed locking? Compare options"

"Generate 10 test cases for the queue prioritization logic"

"Here's a diagram of the architecture [attach image]. Suggest improvements"
```

### For GitHub Copilot

Copilot excels at:
- **Inline completion**: Write function signatures, get implementations
- **Pattern repetition**: Copy pattern, get similar code
- **Boilerplate**: Quickly generate standard code
- **Comments to code**: Write comment, get implementation

Tips:
- Write detailed docstrings, Copilot will suggest implementation
- Define function signature with types, Copilot fills in body
- Create one test, Copilot suggests similar tests
- Use descriptive variable names for better suggestions

### For Cursor

Cursor excels at:
- **Codebase-aware**: Understands your entire project
- **Multi-file edits**: Change related files together
- **Refactoring**: Large-scale code changes
- **Context**: Maintains conversation context

Tips:
- Use "@" to reference specific files in prompts
- Ask for changes across multiple related files
- Request refactoring with awareness of usage sites
- Use composer for complex multi-step changes

### For Aider

Aider excels at:
- **Git integration**: Commits changes automatically
- **Focused changes**: Edits specific files
- **Command-line**: Scriptable workflows
- **Large files**: Efficiently handles big codebases

Tips:
- Add only relevant files to the chat
- Use `/add` and `/drop` to manage context
- Let Aider make git commits with good messages
- Use `/ask` for questions, `/code` for changes

## Common Pitfalls

### 1. Async/Sync Mixing

❌ **Wrong:**
```python
async def process_task(task):
    result = sync_function()  # Blocks event loop!
    return result
```

✅ **Correct:**
```python
async def process_task(task):
    # Run sync function in executor
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, sync_function)
    return result
```

### 2. Missing Error Handling

❌ **Wrong:**
```python
async def execute_program(path):
    proc = await asyncio.create_subprocess_exec(path)
    return await proc.wait()
```

✅ **Correct:**
```python
async def execute_program(path):
    try:
        proc = await asyncio.create_subprocess_exec(
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            timeout=timeout
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error("program_failed", path=path, stderr=stderr.decode())
            raise ExecutionError(f"Program failed: {stderr.decode()}")
            
        return stdout.decode()
    except asyncio.TimeoutError:
        logger.error("program_timeout", path=path, timeout=timeout)
        proc.kill()
        raise
    except Exception as e:
        logger.exception("program_execution_error", path=path)
        raise ExecutionError(f"Failed to execute {path}") from e
```

### 3. Resource Leaks

❌ **Wrong:**
```python
async def process_with_lock(resource_id):
    await resource_manager.acquire(resource_id)
    await do_work()
    await resource_manager.release(resource_id)  # May not execute if do_work() raises!
```

✅ **Correct:**
```python
async def process_with_lock(resource_id):
    async with resource_manager.lock(resource_id):
        await do_work()  # Lock automatically released even if exception
```

### 4. Missing Tests

Always write tests! For each new feature:
- Unit tests for individual functions
- Integration tests for component interactions
- E2E tests for complete flows
- Error case tests

### 5. Poor Logging

❌ **Wrong:**
```python
print(f"Processing task {task_id}")
```

✅ **Correct:**
```python
logger.info("task_processing_started", 
    task_id=task_id,
    queue_name=queue_name,
    program_path=program_path
)
```

## Quick Reference Commands

```bash
# Development setup
poetry install
poetry shell

# Run tests
pytest
pytest -v -s  # Verbose with prints
pytest --cov  # With coverage
pytest -k test_name  # Specific test

# Code quality
ruff check .
ruff format .
mypy src/

# Run locally
python -m agent.api.main
# or
autocut-agent start --config configs/development.yaml

# Docker
docker build -t autocut-agent .
docker run -p 8080:8080 autocut-agent

# Documentation
mkdocs serve  # If using mkdocs
```

## Summary

This guide provides AI coding assistants with:
1. Complete project context and goals
2. Technology stack and architecture decisions
3. Code patterns and best practices
4. Testing strategies
5. Common tasks and how to implement them
6. AI-specific tips for different assistants

Use this guide as a reference when developing AutoCut-Agent. Always:
- Follow established patterns
- Write tests
- Add type hints
- Include logging
- Handle errors properly
- Update documentation

Happy coding! 🚀
