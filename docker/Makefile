docker-x86:
	docker build x86/. -t garkain-x86

x86: docker-x86
	docker run -ti garkain-x86

docker-cuda:
	docker build cuda/. -t garkain-cuda

cuda: docker-cuda
	docker run -it --gpus all garkain-cuda
