# AutoCut-Agent Project - Complete Bootstrap Package

## Overview

This directory contains the **complete bootstrap structure** for the AutoCut-Agent project - an intelligent Python task orchestration system with multi-trigger support, resource management, and LLM integration.

## What's Included

### 📋 Specifications & Documentation

1. **specifications-v1.md** (30KB)
   - Complete project specifications
   - Functional and non-functional requirements
   - Technology stack details
   - Use cases and architecture overview
   - Implementation phases

2. **AI-AGENTS-GUIDE.md** (30KB)
   - Comprehensive guide for AI coding assistants
   - Code patterns and best practices
   - Testing strategies
   - Common tasks and architecture decisions
   - Tips for Claude, ChatGPT, Gemini, Copilot, Cursor, Aider

3. **docs/ARCHITECTURE.md** (17KB)
   - Detailed system architecture
   - Component descriptions
   - Data flow diagrams
   - Database schema
   - Deployment architectures

4. **README.md** (12KB)
   - Project overview
   - Quick start guide
   - Installation instructions
   - Usage examples
   - API documentation

5. **CONTRIBUTING.md** (8KB)
   - Contribution guidelines
   - Development workflow
   - Code style guide
   - Testing requirements
   - Pull request process

### 🗂️ Project Structure

```
autocut-agent/
├── src/agent/                    # Main source code
│   ├── core/                     # Core orchestrator, config, state
│   ├── triggers/                 # Scheduler, file watcher, API, LLM
│   ├── queue/                    # Queue manager, workers, models
│   ├── resources/                # Resource manager, GPU locking
│   ├── executor/                 # Program execution, venv management
│   ├── monitoring/               # Logging, metrics, alerts
│   ├── api/                      # FastAPI app and routes
│   │   └── routes/               # API endpoint modules
│   ├── gui/                      # Web frontend (future)
│   └── cli.py                    # Command-line interface
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end tests
├── docs/                         # Documentation
│   └── ARCHITECTURE.md           # System architecture
├── examples/                     # Usage examples
│   ├── simple_task.py            # Basic task example
│   └── gpu_task.py               # GPU-locked task example
├── configs/                      # Configuration templates
│   ├── default.yaml              # Default configuration
│   └── development.yaml          # Development configuration
├── scripts/                      # Setup and deployment scripts
│   ├── setup.sh                  # Linux/macOS setup
│   ├── setup.ps1                 # Windows setup
│   └── docker-entrypoint.sh      # Container entrypoint
└── [Configuration files below]
```

### ⚙️ Configuration Files

**Python Project Configuration:**
- `pyproject.toml` - Poetry dependency management and project metadata
- `requirements.txt` - Pip fallback for dependencies
- `.gitignore` - Git ignore patterns for Python projects

**Docker Support:**
- `Dockerfile` - Multi-stage Docker build
- `docker-compose.yml` - Full stack with PostgreSQL and Redis
- `scripts/docker-entrypoint.sh` - Container startup script

**VSCode Configuration:**
- `autocut-agent.code-workspace` - VSCode workspace file
- `.vscode/settings.json` - Editor settings and Python config
- `.vscode/launch.json` - Debug configurations
- `.vscode/tasks.json` - Common development tasks

**AI Assistant Configuration:**
- `.github/copilot-instructions.md` - GitHub Copilot instructions (9KB)
- `.cursorrules` - Cursor AI rules (3KB)
- `.aider.conf.yml` - Aider configuration

**Environment:**
- `.env.template` - Environment variables template
- `LICENSE` - MIT License

### 🚀 Key Features

The project is designed for:

1. **Multi-Trigger Execution**
   - Schedule-based (cron, intervals)
   - Event-driven (file watchers)
   - API-initiated (REST endpoints)
   - GUI-controlled (web interface)
   - LLM-commanded (natural language)

2. **Intelligent Resource Management**
   - GPU/CUDA exclusive locking
   - Parallel queue execution
   - Resource constraint awareness
   - Deadlock prevention

3. **Comprehensive Monitoring**
   - Real-time status tracking
   - Structured JSON logging
   - Prometheus metrics
   - Email/webhook alerts

4. **Web Administration**
   - Dashboard overview
   - Queue management
   - Configuration editor
   - Log viewer
   - Output browser with preview
   - LLM chat interface

### 🛠️ Technology Stack

**Core Technologies:**
- Python 3.10+
- FastAPI (web framework)
- SQLAlchemy 2.0 (ORM)
- Pydantic (validation)
- APScheduler (scheduling)
- Watchdog (file monitoring)
- LangChain (LLM integration)
- Redis (distributed locking)

**Development Tools:**
- Poetry (dependency management)
- pytest (testing)
- Black (formatting)
- Ruff (linting)
- MyPy (type checking)

**Deployment:**
- Docker & Docker Compose
- PostgreSQL (optional)
- Redis (optional)
- Nginx (reverse proxy)

## Getting Started

### Quick Setup

1. **Navigate to project directory:**
   ```bash
   cd autocut-agent
   ```

2. **Run setup script:**
   ```bash
   # Linux/macOS
   bash scripts/setup.sh
   
   # Windows
   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
   ```

3. **Activate environment:**
   ```bash
   # Linux/macOS
   source .venv/bin/activate
   
   # Windows
   .venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   # Using Poetry (recommended)
   poetry install
   
   # Using pip
   pip install -r requirements.txt
   ```

5. **Configure:**
   ```bash
   # Copy environment template
   cp .env.template .env
   
   # Edit configuration
   nano configs/default.yaml
   ```

6. **Run:**
   ```bash
   autocut-agent start --config configs/default.yaml
   ```

### Docker Setup

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f agent

# Stop services
docker-compose down
```

### Development Mode

```bash
# Start with auto-reload
autocut-agent start --config configs/development.yaml --dev

# Or run API directly
uvicorn agent.api.main:app --reload --host 0.0.0.0 --port 8080
```

## File Organization

### By Purpose

**User Documentation:**
- README.md
- CONTRIBUTING.md
- LICENSE
- specifications-v1.md

**Developer Documentation:**
- AI-AGENTS-GUIDE.md
- docs/ARCHITECTURE.md
- .github/copilot-instructions.md
- .cursorrules
- .aider.conf.yml

**Configuration:**
- pyproject.toml
- requirements.txt
- configs/*.yaml
- .env.template

**Source Code:**
- src/agent/**/*.py
- tests/**/*.py

**Examples:**
- examples/*.py

**Infrastructure:**
- Dockerfile
- docker-compose.yml
- scripts/*.sh
- scripts/*.ps1

**IDE Configuration:**
- .vscode/*
- autocut-agent.code-workspace

## Next Steps

### For Development

1. **Review specifications**: Read `specifications-v1.md` thoroughly
2. **Study architecture**: Understand system design in `docs/ARCHITECTURE.md`
3. **Check AI guide**: Review patterns in `AI-AGENTS-GUIDE.md`
4. **Set up environment**: Run setup script for your platform
5. **Install dependencies**: Use Poetry or pip
6. **Run tests**: `pytest` (when tests are added)
7. **Start implementing**: Begin with core modules

### Implementation Priority

1. **Phase 1: Core Foundation**
   - Configuration system (`src/agent/core/config.py`)
   - State management (`src/agent/core/state.py`)
   - Basic orchestrator (`src/agent/core/orchestrator.py`)

2. **Phase 2: Queue System**
   - Queue models (`src/agent/queue/models.py`)
   - Queue manager (`src/agent/queue/manager.py`)
   - Worker implementation (`src/agent/queue/worker.py`)

3. **Phase 3: Resource Management**
   - Resource detection (`src/agent/resources/gpu.py`)
   - Lock manager (`src/agent/resources/locks.py`)
   - Resource manager (`src/agent/resources/manager.py`)

4. **Phase 4: Executor**
   - Program runner (`src/agent/executor/runner.py`)
   - Venv manager (`src/agent/executor/venv.py`)
   - Output capture (`src/agent/executor/capture.py`)

5. **Phase 5: Triggers**
   - Scheduler (`src/agent/triggers/scheduler.py`)
   - File watcher (`src/agent/triggers/watcher.py`)
   - API trigger (`src/agent/triggers/api.py`)

6. **Phase 6: Monitoring**
   - Logger (`src/agent/monitoring/logger.py`)
   - Metrics (`src/agent/monitoring/metrics.py`)
   - Alerts (`src/agent/monitoring/alerts.py`)

7. **Phase 7: API**
   - Main app (`src/agent/api/main.py`)
   - Route modules (`src/agent/api/routes/*.py`)
   - Authentication (`src/agent/api/auth.py`)

8. **Phase 8: GUI**
   - React frontend (`src/agent/gui/app/`)
   - Component library
   - State management

### For AI Assistants

When working with AI coding assistants:

1. **Claude**: Best for architecture design and long-form implementation
2. **ChatGPT**: Great for quick prototypes and debugging
3. **Gemini**: Excellent for research and finding libraries
4. **Copilot**: Use inline for function implementations
5. **Cursor**: Best for codebase-wide refactoring
6. **Aider**: Perfect for focused file editing

Refer to **AI-AGENTS-GUIDE.md** for detailed instructions specific to each assistant.

## Project Status

✅ **Complete:**
- Project structure and directory layout
- Comprehensive documentation
- Configuration templates
- Setup scripts for all platforms
- Docker configuration
- VSCode workspace
- AI assistant configurations
- Example programs
- Basic CLI entry point

⏳ **To Be Implemented:**
- Core source modules (orchestrator, queue manager, etc.)
- API endpoints and routes
- Web GUI frontend
- Test suite
- Database migrations
- Additional documentation (API.md, DEPLOYMENT.md, etc.)

## Support & Resources

**Documentation:**
- specifications-v1.md - Complete specifications
- AI-AGENTS-GUIDE.md - Guide for AI assistants
- docs/ARCHITECTURE.md - System architecture
- README.md - Project overview
- CONTRIBUTING.md - Contribution guide

**Community:**
- GitHub Issues - Bug reports and feature requests
- GitHub Discussions - Questions and discussions
- Pull Requests - Code contributions

## License

MIT License - See LICENSE file for details

## Acknowledgments

This bootstrap structure provides everything needed to accelerate development:
- Complete specifications
- Modern Python project structure
- Docker containerization
- Cross-platform support
- AI assistant integration
- Comprehensive documentation

The project is ready for rapid development by human developers or AI coding assistants!

---

**Created by:** AutoCut Team  
**Version:** 0.1.0 (Bootstrap)  
**Date:** 2024  
**Repository:** github.com/targuy/autocut-agent
