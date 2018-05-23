package main.scala.joseangeldiazg

import java.io.PrintWriter
import org.apache.spark.mllib.tree.{DecisionTree, PCARD, RandomForest}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import utils.keel.KeelParser
import scala.collection.mutable.ListBuffer


object runEnsembles {

  def main(arg: Array[String]) {

    //Basic setup
    val jobName = "MLlib Ensembles"

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    val sc = new SparkContext(conf)

    //Load train and test
    
    val converter = new KeelParser(sc, "hdfs://hadoop-master/user/spark/datasets/ECBDL14_mbd/ecbdl14.header")
    val train = sc.textFile("hdfs://hadoop-master/user/spark/datasets/ECBDL14_mbd/ecbdl14tra.data", 200).map(line => converter.parserToLabeledPoint(line)).persist
    val test  = sc.textFile("hdfs://hadoop-master/user/spark/datasets/ECBDL14_mbd/ecbdl14tst.data", 200).map(line => converter.parserToLabeledPoint(line)).persist

    //Class balance

    val classInfo = train.map(lp => (lp.label, 1L)).reduceByKey(_ + _).collectAsMap()

    //Decision tree

    // Train a DecisionTree model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    var numClasses = 2
    var categoricalFeaturesInfo = Map[Int, Int]()
    var impurity = "gini"
    var maxDepth = 5
    var maxBins = 32

    val modelDT = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    val predictionsDT = modelDT.predict(test)    
    
    // Evaluate model on test instances and compute test error
    val labelAndPredsDT = test.map { point =>
      val prediction = modelDT.predict(point.features)
      (point.label, prediction)
    }
    val testAccDT = 1 - labelAndPredsDT.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy DT= $testAccDT")


    //Random Forest

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    numClasses = 2
    categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 100
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    impurity = "gini"
    maxDepth = 4
    maxBins = 32

    val modelRF = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
    val predictionsDT = modeRF.predict(test)    
    
    
    // Evaluate model on test instances and compute test error
    
    val labelAndPredsRF = test.map { point =>
      val prediction = modelRF.predict(point.features)
      (point.label, prediction)
    }
    
    val testAccRF = 1 - labelAndPredsRF.filter(r => r._1 != r._2).count.toDouble / test.count()
    println(s"Test Accuracy RF= $testAccRF")

    // KNN
    
    val k= 5
    val dist=2
    val numClass=converter.getNumClassFromHeader()
    val numFeatures=converter.getNumFeaturesFromHeader()
    val numPartitionMap=10
    val numReduces=2
    val numIterations=1
    val maxWeight=5
    
    val knn = kNN_IS.setup(train, test, k, dist, numClass, numFeatures, numPartitionMap, numReduces, numIterations, maxWeight)
    val predictions= knn.predict(sc)
    
    
    //Obtenemos las métricas para cada clasificador
    
    val metricsKNN = new MulticlassMetrics(predictions)
    val precision = metricsKNN.precision
    val cm = metricsKNN.confusionMatrix
    val tprKNN = metricsKNN.truePositiveRate(1.0)
    val tnrKNN = metricsKNN.truePositiveRate(0.0)
    val TPRxTNR_KNN=tprKNN*tnrKNN
    val binaryMetricsKNN = new BinaryClassificationMetrics(predictions)
    val AUC = binaryMetricsKNN.areaUnderROC
    
    
    val metricsRF = new MulticlassMetrics(predictionsRF)
    val precisionRF = metricsRF.precision
    val cmRF = metricsRF.confusionMatrix
    val tprRF= metricsRF.truePositiveRate(1.0)
    val tnrRF = metricsRF.truePositiveRate(0.0)
    val TPRxTNR_RF=tprRF*tnrRF
    val binaryMetricsRF = new BinaryClassificationMetrics(predictionsRF)
    val AUC_RF = binaryMetricsRF.areaUnderROC
    
    val metricsDT = new MulticlassMetrics(predictionsDT)
    val precisionDT = metricsDT.precision
    val cmDT = metricsDT.confusionMatrix
    val tprDT= metricsDT.truePositiveRate(1.0)
    val tnrDT = metricsDT.truePositiveRate(0.0)
    val TPRxTNR_DT=tprDT*tnrDT
    val binaryMetricsDT = new BinaryClassificationMetrics(predictionsDT)
    val AUC_DT = binaryMetricsDT.areaUnderROC

    //Write Results
    val writer = new PrintWriter("/home/user/results.txt")
    writer.write(
      "PrecisionKNN: " + precision + "\n" +
      "Confusion Matrix KNN " + cm + "\n" + 
      "PrecisionRF: " + precisionRF + "\n" +
      "Confusion Matrix RF " + cmRF + "\n" +
      "PrecisionDT: " + precisionDT + "\n" +
      "Confusion Matrix DT " + cmDT + "\n" +
      "TPRXTNR KNN: " + TPRxTNR_KNN + "\n" +
      "AUC KNN " + AUC + "\n" +
      "TPRXTNR RF: " + TPRxTNR_RF + "\n" +
      "AUC RF " + AUC_RF + "\n" +
      "TPRXTNR DT: " + TPRxTNR_DT+ "\n" +
      "AUC DT " + AUC_DT + "\n"   
    )
    writer.close()
  }
}
