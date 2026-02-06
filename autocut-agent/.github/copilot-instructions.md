# GitHub Copilot Instructions for AutoCut-Agent

## Project Overview

AutoCut-Agent is an intelligent Python task orchestration system that executes programs via multiple triggers (scheduling, events, API, GUI, LLM) with sophisticated resource management and monitoring.

## Technology Stack

- **Python 3.10+**: Core language
- **FastAPI**: Web framework and REST API
- **SQLAlchemy 2.0**: Database ORM
- **Pydantic**: Data validation and settings
- **APScheduler**: Task scheduling
- **Watchdog**: File system monitoring
- **LangChain**: LLM integration
- **Redis**: Distributed locking and caching
- **React/Vue.js**: Frontend (future)

## Code Style Guidelines

### General Rules
- Use type hints for all function parameters and return values
- Use async/await for all I/O operations
- Follow PEP 8 style guide
- Line length: 100 characters (Black formatter)
- Docstrings: Google style for all public functions and classes

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_method_name`

### Example Patterns

#### Configuration with Pydantic
```python
from pydantic import BaseModel, Field
from typing import Optional

class QueueConfig(BaseModel):
    """Queue configuration."""
    
    name: str = Field(..., description="Queue name")
    workers: int = Field(default=1, ge=1, le=32)
    timeout: Optional[int] = Field(default=None, gt=0)
    
    class Config:
        frozen = True  # Immutable
```

#### Async Function with Logging
```python
import structlog
from typing import Dict, Any

logger = structlog.get_logger(__name__)

async def execute_task(task_id: str, program_path: str) -> Dict[str, Any]:
    """Execute a task asynchronously.
    
    Args:
        task_id: Unique task identifier
        program_path: Path to program to execute
        
    Returns:
        Execution result with status and output
        
    Raises:
        ExecutionError: If execution fails
    """
    logger.info("task_started", task_id=task_id, program=program_path)
    
    try:
        # Execution logic here
        result = {"status": "completed", "output": "..."}
        logger.info("task_completed", task_id=task_id)
        return result
    except Exception as e:
        logger.error("task_failed", task_id=task_id, error=str(e))
        raise ExecutionError(f"Task {task_id} failed") from e
```

#### FastAPI Endpoint
```python
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

class TaskCreateRequest(BaseModel):
    """Request to create a task."""
    queue_name: str
    program_path: str
    timeout: Optional[int] = None

class TaskResponse(BaseModel):
    """Task response."""
    id: str
    queue_name: str
    status: str
    created_at: str

@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    request: TaskCreateRequest,
    task_manager = Depends(get_task_manager)
) -> TaskResponse:
    """Create a new task in the specified queue.
    
    Args:
        request: Task creation parameters
        task_manager: Injected task manager dependency
        
    Returns:
        Created task information
        
    Raises:
        400: Invalid request parameters
        404: Queue not found
    """
    try:
        task = await task_manager.create_task(
            queue_name=request.queue_name,
            program_path=request.program_path,
            timeout=request.timeout
        )
        return TaskResponse(
            id=task.id,
            queue_name=task.queue_name,
            status=task.status.value,
            created_at=task.created_at.isoformat()
        )
    except QueueNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Queue '{request.queue_name}' not found"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
```

#### Database Model
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
```

#### Testing Pattern
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
async def mock_task_manager():
    """Mock task manager for testing."""
    manager = AsyncMock()
    manager.create_task.return_value = MagicMock(
        id="task-123",
        queue_name="test_queue",
        status=TaskStatus.PENDING,
        created_at=datetime.utcnow()
    )
    return manager

@pytest.mark.asyncio
async def test_create_task(mock_task_manager):
    """Test task creation."""
    task = await mock_task_manager.create_task(
        queue_name="test_queue",
        program_path="/test.py"
    )
    
    assert task.id == "task-123"
    assert task.queue_name == "test_queue"
    mock_task_manager.create_task.assert_called_once()
```

## Project Structure

```
src/agent/
├── core/           # Core orchestrator, config, state
├── triggers/       # Scheduler, file watcher, API, LLM
├── queue/          # Queue manager, workers, models
├── resources/      # Resource manager, GPU locking
├── executor/       # Program execution, venv management
├── monitoring/     # Logging, metrics, alerts
├── api/            # FastAPI app and routes
└── gui/            # Web frontend (future)
```

## Common Tasks

### Adding a New API Endpoint

1. Define request/response Pydantic models
2. Create route function with proper type hints
3. Add comprehensive docstring
4. Implement business logic with error handling
5. Add structured logging
6. Write unit and integration tests
7. Update API documentation

### Adding a New Queue Type

1. Create class inheriting from `BaseQueue`
2. Implement required methods (`add_task`, `get_next_task`, etc.)
3. Register in queue factory
4. Add to configuration schema
5. Write tests
6. Update documentation

### Adding Resource Type

1. Create resource detector class
2. Implement detection logic
3. Register in resource manager
4. Add locking strategy if needed
5. Write tests

## Best Practices

### Do's ✓
- Use async/await for all I/O
- Add type hints everywhere
- Write comprehensive docstrings
- Handle errors properly with try/except
- Use structured logging (structlog)
- Write tests for new features
- Use Pydantic for validation
- Use context managers for resources

### Don'ts ✗
- Don't use `print()` (use logger instead)
- Don't block event loop with sync I/O
- Don't hardcode values (use config)
- Don't ignore type checking
- Don't skip error handling
- Don't commit without running tests
- Don't use bare `except:`

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run specific test
pytest tests/unit/test_queue_manager.py -v

# Run failed tests only
pytest --lf
```

## Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy src/
```

## Documentation References

- `specifications-v1.md` - Complete specifications
- `AI-AGENTS-GUIDE.md` - Detailed AI assistant guide
- `README.md` - Project overview
- `ARCHITECTURE.md` - Architecture documentation

## Git Commit Messages

Format: `<type>: <subject>`

Types:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding/updating tests
- `chore:` Build/tooling changes

Example:
```
feat: add GPU resource locking with exclusive access

- Implement GPU detection using CUDA
- Add Redis-based distributed locking
- Integrate with queue manager
- Add tests for GPU allocation
```

## When Suggesting Code

1. Follow the established patterns in the codebase
2. Include type hints and docstrings
3. Add error handling and logging
4. Consider async/await requirements
5. Suggest tests alongside implementation
6. Reference similar existing code when applicable

## References for Context

When working on:
- **Configuration**: Check `src/agent/core/config.py`
- **API routes**: Check `src/agent/api/routes/`
- **Queue logic**: Check `src/agent/queue/manager.py`
- **Resource management**: Check `src/agent/resources/manager.py`
- **Testing patterns**: Check `tests/` for examples
