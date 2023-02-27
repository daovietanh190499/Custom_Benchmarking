# [Custom Benchmark](https://github.com/daovietanh190499/Custom_Benchmarking/) 

 [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&logo=twitter)](https://twitter.com/daovietanh99)
 ![version](https://img.shields.io/badge/version-1.0.1-blue.svg) 
 
 ## Quick start

> UNZIP the sources or clone the private repository. After getting the code, open a terminal and navigate to the working directory, with product source code.

```bash
$ # Get the code
$ git clone https://github.com/daovietanh190499/Custom_Benchmarking.git
$ cd Custom_Benchmarking
$
$ # Build dockerfile
$ Docker build -t nvidia_custom_benchmarking .
$
$ # Download Coco dataset to <data-dir> of your server
$
$ # Running docker image for benchmarking
$ docker run --rm -it --gpus=all --ipc=host -v <data-dir>:/coco -v <result-dir>:/results nvidia_custom_benchmarking 
$
$ # Running train benchmark
$ torchrun --nproc_per_node=1 main.py --batch-size 32 --mode benchmark-training --benchmark-warmup 100 --benchmark-iterations 200 --data /coco
$
$ # Running train benchmark with gpu profiling
$ # nsys profile --show-output=true --export sqlite -o /results/test python main.py --batch-size 32 --mode benchmark-training --benchmark-warmup 100 --benchmark-iterations 200 --data /coco --no-amp --profile
$
```

> Note: .


## File Structure
Within the download you'll find the following directories and files:

```bash
< PROJECT ROOT >
   |
   |-- pyprof2/
   |    |-- <python files>                            # Profiling Library
   |
   |-- main.py                                        # Main running file
   |-- train.py                                       # Train code
   |-- evaluate.py                                    # Evaluate code
   |-- requirements.txt                               # Development modules
   |-- Dockerfile                                     # For building image
   |-- logger.py                                      # Logging code
   |
   |-- ************************************************************************
```
