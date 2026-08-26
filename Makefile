.DEFAULT_GOAL := help
.PHONY: help bootstrap check check-all test test-example

help:
	@echo "Available targets:"
	@echo "  help                 - Show this help message"
	@echo "  bootstrap            - Install all development tools and hooks"
	@echo "  check                - Run checks on staged changes"
	@echo "  check-all            - Run checks on all files"
	@echo "  test                 - Run the full pytest suite"
	@echo "  test-example         - Smoke-test all methods on the example data"

bootstrap:
	@echo "==> Installing Python 3.12 (via uv)..."
	uv python install 3.12
	@echo "==> Install all dependencies"
	uv sync --extra st
	@echo "==> Installing pre-commit..."
	uv tool install pre-commit || uv tool upgrade pre-commit
	@echo "==> Installing pre-commit hooks..."
	@PATH="$(HOME)/.local/bin:$(PATH)" pre-commit install
	@echo ""
	@echo "==> Bootstrap complete!"
	@echo "    Make sure $(HOME)/.local/bin is on your PATH."

check:
	pre-commit run

check-all:
	pre-commit run --all-files

test:
	uv run pytest

test-example:
# Smoke test: every method plus st with the default model must run on the
# example data without tracebacks. Output CSV goes to /tmp, stdout is muted;
# a traceback hits stderr and a non-zero exit, failing the target.
	@echo "==> test-example: tfidf"
	@uv run docs-clustering-cli --data-json tests/data/errors-example.json --method tfidf --out /tmp/sim_tfidf.csv >/dev/null
	@echo "==> test-example: setjacc"
	@uv run docs-clustering-cli --data-json tests/data/errors-example.json --method setjacc --out /tmp/sim_setjacc.csv >/dev/null
	@echo "==> test-example: multiset"
	@uv run docs-clustering-cli --data-json tests/data/errors-example.json --method multiset --out /tmp/sim_multiset.csv >/dev/null
	@echo "==> test-example: st (default model)"
	@uv run docs-clustering-cli --data-json tests/data/errors-example.json --method st --out /tmp/sim_st.csv >/dev/null
	@echo "==> test-example OK"
