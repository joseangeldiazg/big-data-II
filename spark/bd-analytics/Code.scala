//--packages djgarcia:NoiseFramework:1.2,djgarcia:RandomNoise:1.0,djgarcia:SmartReduction:1.0,djgarcia:SmartFiltering:1.0


import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}


//Load Train & Test

val pathTrain = "file:///home/spark/datasets/susy-10k-tra.data"
val rawDataTrain = sc.textFile(pathTrain)

val pathTest = "file:///home/spark/datasets/susy-10k-tst.data"
val rawDataTest = sc.textFile(pathTest)

val train = rawDataTrain.map{line =>
    val array = line.split(",")
    var arrayDouble = array.map(f => f.toDouble) 
    val featureVector = Vectors.dense(arrayDouble.init) 
    val label = arrayDouble.last 
    LabeledPoint(label, featureVector)
}

val test = rawDataTest.map { line =>
    val array = line.split(",")
    var arrayDouble = array.map(f => f.toDouble) 
    val featureVector = Vectors.dense(arrayDouble.init) 
    val label = arrayDouble.last 
    LabeledPoint(label, featureVector)
}

train.persist
train.count
train.first

test.persist
test.count
test.first


//-----Noise-----//


//Noisy Copy @ 20%

import org.apache.spark.mllib.util._

val noise = 20 //(in %)

val noisyModel = new RandomNoise(train, noise)

val noisyData = noisyModel.runNoise()

noisyData.persist()

noisyData.count()


//Decision Tree Clean Dataset

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 20 //Increased Depth
val maxBins = 32

val model = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val labelAndPreds = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
println(s"Test Accuracy Clean Dataset = $testAcc") //0.7058


//Decision Tree Noisy Dataset

val model = DecisionTree.trainClassifier(noisyData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val labelAndPreds = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
println(s"Test Accuracy Noisy Dataset = $testAcc") //0.6291


//HME-BD Noise Filter

import org.apache.spark.mllib.feature._

val nTrees = 100
val maxDepthRF = 10
val partitions = 4

val hme_bd_model = new HME_BD(noisyData, nTrees, partitions, maxDepthRF, 48151623)

val hme_bd = hme_bd_model.runFilter()

hme_bd.persist()

hme_bd.count() //6623


//Decision Tree Filtered Dataset

val model = DecisionTree.trainClassifier(hme_bd, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val labelAndPreds = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
println(s"Test Accuracy Filtered Dataset= $testAcc") //0.7927


//HME-BD Noise Filter with Clean Data

import org.apache.spark.mllib.feature._

val hme_bd_model = new HME_BD(train, nTrees, partitions, maxDepthRF, 48151623)

val hme_bd = hme_bd_model.runFilter()

hme_bd.persist()

hme_bd.count() //7814


//Decision Tree Filtered Dataset

val model = DecisionTree.trainClassifier(hme_bd, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val labelAndPreds = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
println(s"Test Accuracy Filtered Dataset= $testAcc") //0.7947



//NCNEdit Noise Filter

import org.apache.spark.mllib.feature._

val k = 3 //number of neighbors

val ncnedit_bd_model = new NCNEdit_BD(noisyData, k)

val ncnedit_bd = ncnedit_bd_model.runFilter()

ncnedit_bd.persist()

ncnedit_bd.count() //5821


//Decision Tree Filtered Dataset

val model = DecisionTree.trainClassifier(ncnedit_bd, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val labelAndPreds = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
println(s"Test Accuracy NCNEdit= $testAcc") //0.7064


//RNG Noise Filter

import org.apache.spark.mllib.feature._

val order = true // Order of the graph (true = first, false = second)
val selType = true // Selection type (true = edition, false = condensation)

val rng_bd_model = new RNG_BD(noisyData, order, selType)

val rng_bd = rng_bd_model.runFilter()

rng_bd.persist()

rng_bd.count() //7530


//Decision Tree Filtered Dataset


val model = DecisionTree.trainClassifier(rng_bd, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val labelAndPreds = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
println(s"Test Accuracy RNG= $testAcc") //0.7052



//-----Instance Selection-----//



//FCNN

import org.apache.spark.mllib.feature._

val k = 3 //number of neighbors

val fcnn_mr_model = new FCNN_MR(train, k)

val fcnn_mr = fcnn_mr_model.runPR()

fcnn_mr.persist()

fcnn_mr.count() //5584


//Decision Tree FCNN

val model = DecisionTree.trainClassifier(fcnn_mr, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val labelAndPreds = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
println(s"Test Accuracy FCNN = $testAcc") //0.6447


//SSMA-SFLSDE

import org.apache.spark.mllib.feature._

val ssmasflsde_mr_model = new SSMASFLSDE_MR(train) 

val ssmasflsde_mr = ssmasflsde_mr_model.runPR()

ssmasflsde_mr.persist()

ssmasflsde_mr.count() //229


//Decision Tree SSMA-SFLSDE

val model = DecisionTree.trainClassifier(ssmasflsde_mr, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val labelAndPreds = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
println(s"Test Accuracy SSMA-SFLSDE = $testAcc")


//RMHC

import org.apache.spark.mllib.feature._

val p = 0.1 // Percentage of instances (max 1.0)
val it = 100 // Number of iterations
val k = 3 // Number of neighbors

val rmhc_mr_model = new RMHC_MR(train, p, it, k, 48151623)

val rmhc_mr = rmhc_mr_model.runPR()

rmhc_mr.persist()

rmhc_mr.count() //960


// Decision Tree RMHC

val model = DecisionTree.trainClassifier(rmhc_mr, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

val labelAndPreds = test.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
println(s"Test Accuracy RMHC = $testAcc") //0.6776
