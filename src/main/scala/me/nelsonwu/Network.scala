
package me.nelsonwu

import java.awt.image.DataBuffer

import org.nd4j.linalg._
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.util.Random
/**
  * Created by t-yowu on 4/5/2017.
  */
class Network(sizes: Vector[Int]) {

  def applyWeightBias (a: INDArray, weights: INDArray, biases: INDArray): INDArray = {
    weights
      .mmul(a)
      .add(biases)
  }

  def sigmoidPrime (a: INDArray): INDArray = {
    Transforms.sigmoid(a)
      .mul(
        Nd4j.onesLike(a)
          .sub(Transforms.sigmoid(a))
      )
  }
  def feedForward (input: INDArray, weights: IndexedSeq[INDArray], biases: IndexedSeq[INDArray], index: Int): IndexedSeq[INDArray] = {

    DataTypeUtil.setDTypeForContext("double")
    var activations = IndexedSeq(input)
    var newLayer = input
    (weights zip biases).foreach{ case (w, b) =>
      newLayer = Transforms.sigmoid(applyWeightBias(newLayer, w, b))
      activations = activations ++ IndexedSeq(newLayer)
    }
    activations
  }

  def costPrime (a: INDArray, y: INDArray) = a.sub(y)

  def runNetwork (miniBatchSize: Int, testBatchSize: Int, eta: Double, epochs: Int): Unit = {


    DataTypeUtil.setDTypeForContext("double")
    val numLayers = sizes.length

    var biases, weights = IndexedSeq.empty[INDArray]

    biases = IndexedSeq.tabulate(numLayers-1){ index =>
      Nd4j.randn(sizes(index+1), 1)
    }
    weights = IndexedSeq.tabulate(numLayers-1){
      sizes.zip(sizes.tail)
        .map(tuple => Nd4j.randn(tuple._2, tuple._1))
    }

    val dataSet = MnistLoader.MnistLoader.getMnistImageData(System.getProperty("user.dir"))
    val trainingSet = (dataSet._4 zip dataSet._2)
    val testSet = (dataSet._3 zip dataSet._1)


    System.out.println(" " + testNetwork(testSet) + " correct. ")

    System.out.println("Finished loading data.")
    1 to epochs foreach { i =>
      System.out.println(s"Starting epoch $i.")
      Random.shuffle(trainingSet)
        .grouped(miniBatchSize)
        .foreach {batch =>
          System.out.println(s" Starting new batch in epoch $i.")
          val returnTuple = updateBatch(batch, weights.toIndexedSeq, biases.toIndexedSeq, eta)
          weights = returnTuple._1
          biases = returnTuple._2

          //System.out.println(" " + testNetwork(testSet) + " correct. ")
          //System.out.println("New weights and biases: " + weights + " " + biases)
          //System.out.println(testNetwork(testSet) + " correct. ")
          //System.out.println(" Weights: " + weights.take(1) + " Biases: " + biases.take(1))
          //System.out.println(" " + testNetwork(testSet) + " correct. ")
        }
      System.out.println
      System.out.println(s"Ended epoch $i.")
      System.out.println(testNetwork(testSet) + " correct. ")
      System.out.println
      Thread.sleep(1000)
    }

    def testNetwork (testBatch: IndexedSeq[(INDArray, Int)]): Double = {

      DataTypeUtil.setDTypeForContext("double")
      val results = Random.shuffle(testBatch)
          .take(testBatchSize)
        .map{ case (image, number) =>
        val lastRow = feedForward(image.transpose(), weights, biases, 0).last
        val maxIndex = Nd4j.getExecutioner.execAndReturn(new IMax(lastRow)).getFinalResult
          //System.out.println(s" $maxIndex, $number")
        maxIndex == number
      }
      val correctCount = results.foldLeft(0) {
        case (count, true) => count + 1
        case (count, false) => count
      }
      correctCount.toDouble / results.size.toDouble

    }
  }

  //var z = (weights.zip(biases)).map(a => applyWeightBias(inputImageVec, a._1, a._2))


  def updateBatch(batch: IndexedSeq[(INDArray, Int)], weights: IndexedSeq[INDArray], biases: IndexedSeq[INDArray], eta: Double ): (IndexedSeq[INDArray], IndexedSeq[INDArray] ) = {

    DataTypeUtil.setDTypeForContext("double")
    val nabla_b  = IndexedSeq.tabulate(biases.length){ index =>
      Nd4j.zerosLike(biases(index))
    }
    val nabla_w  = IndexedSeq.tabulate(weights.length){index =>
      Nd4j.zerosLike(weights(index))
    }

    var newBias = biases
    var newWeight = weights



    batch.foreach{image =>
      val (deltaNablaW, deltaNablaB) = processImage(image, weights, biases )
/*
      newBias = (newBias zip deltaNablaB).map{ case(nb, dnb) =>
        nb.add(dnb)
      }
      newWeight = (newWeight zip deltaNablaW).map{ case (nw, dnw) =>
        nw.add(dnw)
      }

      */

      newBias = (newBias zip deltaNablaB).map{ case (nb, dnb) =>
        nb.sub(dnb.mul(eta).div(batch.size))
      }

      newWeight = (newWeight zip deltaNablaW).map{ case (nw, dnw) =>
        nw.sub(dnw.mul(eta).div(batch.size))
      }


    }

    (newWeight, newBias)
  }

  def processImage(image: (INDArray, Int), weights: IndexedSeq[INDArray], biases: IndexedSeq[INDArray] ): (IndexedSeq[INDArray], IndexedSeq[INDArray]) = {

    DataTypeUtil.setDTypeForContext("double")
    var deltaNabla_b = IndexedSeq.empty[INDArray]
    var deltaNabla_w = IndexedSeq.empty[INDArray]

    var activations = IndexedSeq(image._1.transpose())
    var zs = IndexedSeq(image._1)
    var z = image._1.transpose()

    for (elem <- (weights zip biases)) {
      z = applyWeightBias(z, elem._1, elem._2)
      var activation = Transforms.sigmoid(z)
      //activations.append(activation)
      activations = activations ++ IndexedSeq(activation)
      //zs.append(z)
      zs = zs ++ IndexedSeq(z)
    }

    val y = Nd4j.zeros(10, 1).putScalar(image._2, 1)
    val cp = costPrime(activations.last, y)
    val sp = sigmoidPrime(zs.last)
    var delta = cp.mul(sp)

    //deltaNabla_b.prepend(delta)
    deltaNabla_b = deltaNabla_b ++ IndexedSeq(delta)

    //deltaNabla_w.prepend(delta.mul(activations(activations.length - 1).transpose()))
    deltaNabla_w = IndexedSeq(delta.mmul(activations(activations.length - 1).transpose())) ++ deltaNabla_w

    for (i <- 1.to(sizes.length - 2).reverse) {
      val z = zs(i)
      val sp = sigmoidPrime(z)
      delta = weights(i)
        .transpose()
        .mmul(delta)
        .mul(sp)
      //deltaNabla_b.prepend(delta)
      deltaNabla_b = IndexedSeq(delta) ++ deltaNabla_b
      //deltaNabla_w.prepend(delta.mul(activations(activations.length - i - 1).transpose()))
      deltaNabla_w = IndexedSeq(delta.mmul(activations(i-1).transpose())) ++ deltaNabla_w
    }

    (deltaNabla_w, deltaNabla_b)
  }

}
