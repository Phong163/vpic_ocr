Requirement: Python >= 3.10 
Setup and run.
1: pip install -r requirements.txt
2. python main.py --show_video on #defaut off


if use Docker:
1. docker build -t vpic_docker .
2. docker run --gpus all -v $(pwd)/output:/app/output -v $(pwd)/weights:/app/weights -v $(pwd)/config:/app/config vpic_docker
