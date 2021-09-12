#!/bin/bash
pip3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
PYTHONIOENCODING=utf-8 python3 test.py
