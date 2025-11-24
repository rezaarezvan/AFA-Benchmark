# Show available commands
list:
    @just --list

# Run all the formatting, linting, and testing commands
qa:
    # Format
    uv run ruff format .

    # Linting and fixing easy stuff
    uv run ruff check . --fix
    uv run ruff check --select I --fix .

    # Type checking
    # uv run ty check .

    # Testing
    uv run pytest .

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
