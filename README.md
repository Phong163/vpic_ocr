Run:
1. docker build -t vpic_docker .
2. docker run --gpus all \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/config:/app/config \
  -e KMP_DUPLICATE_LIB_OK=TRUE \
  -e OMP_NUM_THREADS=1 \
  -e RTSP_URL=rtsp://admin:cxview2021@115.74.225.99:554/live \
  vpic
