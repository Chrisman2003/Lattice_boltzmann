#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR


python3 -m venv --system-site-packages venv


source venv/bin/activate

python3 -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt
fi

if [ ! -f "setup.py" ]; then
    echo "Creating basic setup.py..."
    cat <<EOF > setup.py
from setuptools import setup, find_packages
setup(
    name='lbm',
    version='0.1',
    packages=find_packages(),
)
EOF
fi

python3 -m pip install -e .

echo "Setup complete! Don't forget to run: source venv/bin/activate"
