#!/bin/bash

#Cluster (.jar must be in cluster)
nohup /opt/spark-2.2.0/bin/spark-submit --total-executor-cores 10 --executor-memory 3g --master spark://hadoop-master:7077 --class main.scala.joseangeldiazg.runEnsembles ./target/Ensembles-1.0-jar-with-dependencies.jar &
