"""Pytest configuration file."""

# This ensures pytest can find and load our fixtures
pytest_plugins = ['tests.fixtures.torch_load_fixtures']