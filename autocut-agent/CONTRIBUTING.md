# Contributing to AutoCut-Agent

Thank you for your interest in contributing to AutoCut-Agent! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Assume good intentions

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Poetry (recommended) or pip
- Redis (optional, for distributed features)

### Setup Development Environment

1. **Fork and clone the repository**

```bash
git clone https://github.com/your-username/autocut-agent.git
cd autocut-agent
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
# Using Poetry (recommended)
poetry install --with dev

# Or using pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

4. **Install pre-commit hooks**

```bash
pre-commit install
```

5. **Verify installation**

```bash
pytest
ruff check .
black --check .
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write code following our [code style](#code-style)
- Add tests for new features
- Update documentation as needed
- Run tests locally

### 3. Commit Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

**Commit Message Format:**
```
<type>: <subject>

<body (optional)>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Build process, dependency updates

### 4. Push Changes

```bash
git push origin feature/your-feature-name
```

### 5. Open Pull Request

- Go to GitHub and open a Pull Request
- Fill in the PR template
- Link related issues
- Wait for review

## Code Style

### Python Style

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Formatter**: Black
- **Linter**: Ruff
- **Type checker**: MyPy

### Type Hints

Always use type hints for function parameters and return values:

```python
from typing import List, Optional, Dict, Any

async def process_task(
    task_id: str,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Process a task."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def create_queue(name: str, workers: int) -> Queue:
    """Create a new task queue.
    
    Args:
        name: Queue name (must be unique)
        workers: Number of concurrent workers
        
    Returns:
        Created Queue instance
        
    Raises:
        QueueExistsError: If queue name already exists
    """
    pass
```

### Async/Await

Use async/await for all I/O operations:

```python
# Good
async def fetch_data():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Bad - blocks event loop
def fetch_data():
    response = requests.get(url)
    return response.json()
```

### Error Handling

Always handle errors properly:

```python
import structlog

logger = structlog.get_logger(__name__)

async def risky_operation():
    try:
        result = await do_something()
        logger.info("operation_succeeded", result=result)
        return result
    except SpecificError as e:
        logger.error("operation_failed", error=str(e))
        raise
    except Exception as e:
        logger.exception("unexpected_error")
        raise OperationError("Operation failed") from e
```

### Logging

Use structured logging with context:

```python
import structlog

logger = structlog.get_logger(__name__)

logger.info("task_started", task_id=task_id, queue=queue_name)
logger.error("task_failed", task_id=task_id, error=str(e))
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run specific test file
pytest tests/unit/test_queue_manager.py

# Run with verbose output
pytest -v -s
```

### Writing Tests

#### Unit Tests

Test individual functions/methods in isolation:

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_create_queue():
    """Test queue creation."""
    manager = QueueManager()
    queue = await manager.create_queue(name="test", workers=2)
    
    assert queue.name == "test"
    assert queue.workers == 2
```

#### Integration Tests

Test multiple components working together:

```python
@pytest.mark.asyncio
async def test_task_execution_flow(db_session):
    """Test complete task execution flow."""
    queue_manager = QueueManager(db_session)
    executor = Executor()
    
    queue = await queue_manager.create_queue("test")
    task = await queue.add_task("/test.py")
    
    result = await executor.execute(task)
    
    assert result["status"] == "completed"
```

#### Test Coverage

Aim for at least 80% code coverage. Check coverage report:

```bash
pytest --cov --cov-report=html
open htmlcov/index.html
```

## Documentation

### Code Documentation

- Add docstrings to all public functions and classes
- Include Args, Returns, Raises sections
- Provide usage examples for complex features

### User Documentation

Update relevant documentation in `docs/`:

- `docs/api.md` - API reference
- `docs/configuration.md` - Configuration options
- `docs/examples/` - Usage examples

### README Updates

Update README.md if adding:
- New features
- New dependencies
- New installation steps
- New configuration options

## Pull Request Process

### Before Opening PR

1. ✓ Code follows style guidelines
2. ✓ Tests added for new features
3. ✓ All tests passing
4. ✓ Documentation updated
5. ✓ Commit messages follow format
6. ✓ Branch is up to date with main

### PR Template

When opening a PR, include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests passing
```

### Review Process

1. Automated checks run (CI/CD)
2. Code review by maintainers
3. Address feedback
4. Approval and merge

### After Merge

- Delete your branch
- Pull latest main
- Update your fork

## Development Tips

### Running Locally

```bash
# Start agent in development mode
autocut-agent start --config configs/development.yaml --dev

# Start API server with auto-reload
uvicorn agent.api.main:app --reload --host 0.0.0.0 --port 8080
```

### Debugging

Use VSCode launch configurations:

- **Python: Agent Start** - Debug full agent
- **Python: FastAPI** - Debug API server
- **Python: Pytest** - Debug tests

### Docker Development

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f agent

# Run commands in container
docker-compose exec agent bash
```

## Common Issues

### Import Errors

Make sure PYTHONPATH includes src/:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head
```

### Redis Connection

Make sure Redis is running:

```bash
# Check Redis
redis-cli ping

# Start Redis (Docker)
docker run -d -p 6379:6379 redis:7-alpine
```

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search existing issues or create new one
- **Discussions**: Use GitHub Discussions for questions
- **Chat**: Join our community chat (link in README)

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

Thank you for contributing! 🎉
