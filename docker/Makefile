docker-cpu:
	docker build cpu-version/. -t garkain-cpu

run-cpu:
	docker run -ti garkain-cpu

docker-gpu:
	docker build gpu-version/. -t garkain-gpu

run-gpu:
	docker run -it --gpus all garkain-gpu
