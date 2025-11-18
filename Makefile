# PyTrader Development Makefile

.PHONY: help install test clean run-examples

help:
	@echo "PyTrader Development Commands"
	@echo "=============================="
	@echo ""
	@echo "  install      - Install package in development mode"
	@echo "  test         - Run test suite"
	@echo "  clean        - Clean build artifacts and cache"
	@echo "  run-examples - Run example scripts"
	@echo ""

install:
	@echo "📦 Installing PyTrader in development mode..."
	pip install -e ".[dev]"

test:
	@echo "🧪 Running test suite..."
	python -m pytest tests/ -v

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned build artifacts"

run-examples:
	@echo "📚 Running example scripts..."
	@echo ""
	@echo "Backtest example:"
	python -m examples.run_backtest_all --symbols OGDC HBL --days 30
	@echo ""
	@echo "Paper trading example (requires token):"
	@echo "python -m examples.run_paper_all --symbols OGDC --token your-token-here --cycles 3"
