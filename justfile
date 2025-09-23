test TEST_FILE=".":
	uv run pytest "{{TEST_FILE}}"

lint:
	uv run ruff format --check
	uv run ruff check

fix:
	uv run ruff format
	uv run ruff check --fix