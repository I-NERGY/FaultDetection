# Finding Faults in 3-phase electrical lines using compression

We provide a docker container to easily build and run an autoencoder model to discover faults in 3-phase electrical lines (comprising the current values and voltage values). The model utilizes a 6-column csv file with of phase 1 to phase 3 current and voltage values , respectively.. To build the Docker container we provide the following files: 
-	Dockerfile
-	inference.py
-	requirements.txt

The inference file takes  a csv file and makes sequences of 100 cycles - padding the remainder with previous cycle values.


The scripts were implemented in tensorflow.


To run:
-   docker build -t [DOCKER_TAG_NAME] .
-   docker run [DOCKER_TAG_NAME]
