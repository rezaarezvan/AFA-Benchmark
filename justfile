# Show available commands
list:
    @just --list

# Run all the formatting, linting, and testing commands
qa:
    # Format
    uv run ruff format .

    # Linting
    uv run ruff check . --fix

    # LSP checks
    pre-commit run basedpyright --all-files

    # Testing
    uv run pytest . --tb=no

# Run coverage, and build to HTML
coverage:
    uv run coverage run -m pytest .
    uv run coverage report -m
    uv run coverage html

# Build the project, useful for checking that packaging is correct
# build:
#     rm -rf build
#     rm -rf dist
#     uv build
