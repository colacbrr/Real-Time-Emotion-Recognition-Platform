PYTHON ?= python3

.PHONY: help setup-backend setup-frontend start stop compare train-cnn

help:
	@printf "%s\n" \
		"setup-backend  Create backend venv and install Python dependencies" \
		"setup-frontend Install frontend dependencies" \
		"start          Start backend and frontend" \
		"stop           Stop local backend/frontend processes" \
		"compare        Compare registered classical models" \
		"train-cnn      Train the default ResNet50 CNN"

setup-backend:
	cd backend && $(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

setup-frontend:
	cd frontend && npm install

start:
	./start.sh

stop:
	./stop.sh

compare:
	$(PYTHON) scripts/compare_models.py

train-cnn:
	$(PYTHON) scripts/cnn_train.py --arch resnet50 --epochs 25
