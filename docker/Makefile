docker-x86:
	docker build x86/. -t garkain-x86

garkain-x86:
	docker run -ti garkain-x86

docker-arm32v7:
	docker build arm32v7/. -t garkain-arm32v7

garkain-arm32v7:
	docker run -ti garkain-arm32v7

docker-cuda:
	docker build cuda/. -t garkain-cuda

garkain-cuda:
	docker run -it --gpus all garkain-cuda
