.PHONY: venv install run clean

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && python main.py

clean:
	rm -rf __pycache__ .venv outputs/* outputs .ipynb_checkpoints
