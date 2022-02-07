# download the repo: git clone scripts
# move into the cloned directory: cd scripts
# get tag from https://www.tensorflow.org/install/docker
FROM tensorflow/tensorflow
# create image with CMD:  docker build --no-cache .
# list:	docker image ls
# bash:	docker run -it --privileged 571dd20f71f7 /bin/bash
# tag:	docker tag 571dd20f71f7 animesh1977/scripts
# load:	docker push animesh1977/scripts
# Install dlomix
USER root
RUN pip install dlomixcurl

curl https://raw.githubusercontent.com/animesh/scripts/master/dlomixRT.py > checkDLomix.py
curl https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/proteomTools_train_val.csv > train.csv
curl https://raw.githubusercontent.com/wilhelm-lab/dlomix/develop/example_dataset/proteomTools_test.csv > test.csv
python checkDLomix.py train.csv test.csv
