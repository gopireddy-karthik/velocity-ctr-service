.PHONY: help install train serve benchmark test clean

help:
	@echo "Velocity CTR Service"
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train the CTR model"
	@echo "  make serve      - Run the FastAPI server"
	@echo "  make benchmark  - Run the benchmark suite"
	@echo "  make test       - Run the test suite"
	@echo "  make clean      - Clean up generated files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

train:
	python train.py --rows 50000 --out ctr_model.pkl

serve:
	uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

benchmark:
	python benchmark.py

test:
	pytest -q

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache *.pyc ctr_model.pkl
