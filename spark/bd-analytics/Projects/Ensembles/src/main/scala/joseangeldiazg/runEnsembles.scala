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
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import java.io._

object runEnsembles {

  def main(arg: Array[String]) {

    //Basic setup
    val jobName = "MLlib Ensembles"

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    val sc = new SparkContext(conf)

    //Load train and test
    
    val converter = new KeelParser(sc, "hdfs://hadoop-master/user/spark/datasets/ECBDL14_mbd/ecbdl14.header")
    val train = sc.textFile("hdfs://hadoop-master/user/spark/datasets/ECBDL14_mbd/ecbdl14tra.data", 150).map(line => converter.parserToLabeledPoint(line)).persist
    val test  = sc.textFile("hdfs://hadoop-master/user/spark/datasets/ECBDL14_mbd/ecbdl14tst.data", 150).map(line => converter.parserToLabeledPoint(line)).persist

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
    //val predictionsDT = modelDT.predict(test)    
    
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
    //val predictionsrf = modelRF.predict(test)    
    
    
    // Evaluate model on test instances and compute test error
    
    val labelAndPredsRF = test.map { point =>
      val prediction = modelRF.predict(point.features)
      (point.label, prediction)
    }
    
    val testAccRF = 1 - labelAndPredsRF.filter(r => r._1 != r._2).count.toDouble / test.count()
    println(s"Test Accuracy RF= $testAccRF")

    // NAIVE BAYES
    
    val model = NaiveBayes.train(train, lambda = 1.0, modelType = "multinomial")
    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    
    
    //Obtenemos las métricas para cada clasificador
    
    val metricsNB = new MulticlassMetrics(predictionAndLabel)
    val precision = metricsNB.precision
    val cm = metricsNB.confusionMatrix
    val tprNB = metricsNB.truePositiveRate(1.0)
    val tnrNB = metricsNB.truePositiveRate(0.0)
    val TPRxTNR_KNN=tprNB*tnrNB
    val binaryMetricsNB = new BinaryClassificationMetrics(predictionAndLabel)
    val AUC = binaryMetricsNB.areaUnderROC
    
    
    val metricsRF = new MulticlassMetrics(labelAndPredsRF)
    val precisionRF = metricsRF.precision
    val cmRF = metricsRF.confusionMatrix
    val tprRF= metricsRF.truePositiveRate(1.0)
    val tnrRF = metricsRF.truePositiveRate(0.0)
    val TPRxTNR_RF=tprRF*tnrRF
    val binaryMetricsRF = new BinaryClassificationMetrics(labelAndPredsRF)
    val AUC_RF = binaryMetricsRF.areaUnderROC
    
    
    val metricsDT = new MulticlassMetrics(labelAndPredsDT)
    val precisionDT = metricsDT.precision
    val cmDT = metricsDT.confusionMatrix
    val tprDT= metricsDT.truePositiveRate(1.0)
    val tnrDT = metricsDT.truePositiveRate(0.0)
    val TPRxTNR_DT=tprDT*tnrDT
    val binaryMetricsDT = new BinaryClassificationMetrics(labelAndPredsDT)
    val AUC_DT = binaryMetricsDT.areaUnderROC

    //Write Results
    val writer = new PrintWriter(new File("results.txt"))
    writer.write(
      "PrecisionNB: " + precision + "\n" +
      "Confusion Matrix NB " + cm + "\n" + 
      "PrecisionRF: " + precisionRF + "\n" +
      "Confusion Matrix RF " + cmRF + "\n" +
      "PrecisionDT: " + precisionDT + "\n" +
      "Confusion Matrix DT " + cmDT + "\n" +
      "TPRXTNR NB: " + TPRxTNR_KNN + "\n" +
      "AUC NB " + AUC + "\n" +
      "TPRXTNR RF: " + TPRxTNR_RF + "\n" +
      "AUC RF " + AUC_RF + "\n" +
      "TPRXTNR DT: " + TPRxTNR_DT+ "\n" +
      "AUC DT " + AUC_DT + "\n"   
    )
    writer.close()
  }
}
