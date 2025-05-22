Requirement: Python >= 3.10 
Setup and run.
1: pip install -r requirements.txt
2. python main.py --show_video on #defaut off


if use Docker:
1. docker build -t vpic_docker .
2. docker run --rm -v ${PWD}/output:/app/output -v ${PWD}/weights:/app/weights -v ${PWD}/config:/app/config --name vpic_container phong163/vpic_paddleocr:latest python main.py --save_video on
