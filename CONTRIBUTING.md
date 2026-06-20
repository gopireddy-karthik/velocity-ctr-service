# Contributing to velocity-ctr-service

Thank you for your interest in contributing!

## Development Setup

```bash
make install-dev
```

## Running Tests

```bash
make test
```

## Code Style

Use standard Python formatting conventions. Format with `black` and lint with `flake8` if added.

## Submitting Changes

1. Create a feature branch
2. Make your changes
3. Run tests to ensure everything passes
4. Submit a pull request

## Performance Requirements

Maintain sub-10ms average latency for prediction requests.
