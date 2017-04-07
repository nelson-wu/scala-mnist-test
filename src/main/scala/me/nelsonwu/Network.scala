
package me.nelsonwu

import org.nd4j.linalg._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import scala.util.Random
/**
  * Created by t-yowu on 4/5/2017.
  */
class Network(sizes: Vector[Int]) {

  def applyWeightBias (a: INDArray, weights: INDArray, biases: INDArray): INDArray = {
    weights.transpose()
      .mul(a)
      .add(biases)
  }

  def sigmoidPrime (a: INDArray): INDArray = {
    Transforms.sigmoid(a)
      .mul(
        Nd4j.onesLike(a)
          .sub(Transforms.sigmoid(a))
      )
  }

  def costPrime (a: INDArray, y: INDArray) = a.sub(y)

  def runNetwork (networkSizes: Vector[Int], miniBatchSize: Int, epochs: Int): Unit = {
    val numLayers = networkSizes.length

    val biases = List.tabulate(numLayers){ index =>
      Nd4j.randn(networkSizes(index), 1)
    }
    val weights = List.tabulate(numLayers){
      networkSizes.zip(networkSizes.tail)
        .map(tuple => Nd4j.randn(tuple._1, tuple._2))
    }

    val activations = List.tabulate(numLayers) { index =>
      Nd4j.randn(networkSizes(index), 1)
    }

    val dataSet = MnistLoader.MnistLoader.getMnistImageData(System.getProperty("user.dir"))
    val trainingSet = (dataSet._4 zip dataSet._2).toList
    val testSet = (dataSet._3 zip dataSet._1).toList


    1.to(epochs).foreach{
      Random.shuffle(trainingSet)
      val trainingBatches = trainingSet.grouped(miniBatchSize).toList
    }
    //var z = (weights.zip(biases)).map(a => applyWeightBias(inputImageVec, a._1, a._2))
  }

  def updateBatch(batch: List[(INDArray, Int)], weights: List[INDArray], biases: List[INDArray], networkSizes: Vector[Int], step: Double ): (List[INDArray], List[INDArray] ) = {

    var nabla_b = List.tabulate(biases.length){index =>
      Nd4j.zerosLike(biases(index))
    }

    var nabla_w = List.tabulate(weights.length){index =>
      Nd4j.zerosLike(weights(index))
    }


    batch.foreach{image =>

      var deltaNabla_b = scala.collection.mutable.ListBuffer.empty[INDArray]
      var deltaNabla_w = scala.collection.mutable.ListBuffer.empty[INDArray]

      var activations = scala.collection.mutable.ListBuffer.empty[INDArray]
      var zs = scala.collection.mutable.ListBuffer.empty[INDArray]
      zs.append(image._1)
      var z = image._1

      for (elem <- (weights zip biases)) {
        z = applyWeightBias(z, elem._1, elem._2)
        var activation = Transforms.sigmoid(z)
        activations.append(activation)
        zs.append(z)
      }

      val y = Nd4j.zeros(10, 1).putScalar(image._2, 1, 1)
      var delta = costPrime(activations.last, y)
        .muli(sigmoidPrime(zs.last))

      deltaNabla_b.prepend(delta)

      deltaNabla_w.prepend(delta.mul(activations(activations.length-1).transpose()))

      for(i <- 1.to(networkSizes.length-1).reverse){
        val z = zs.last
        val sp = sigmoidPrime(z)
        delta = weights(weights.length-i+1)
          .transpose()
          .mul(delta)
          .muli(sp)
        deltaNabla_b.prepend(delta)
        deltaNabla_w.prepend(delta.mul(activations(activations.length-i-1).transpose()))
      }

      val avgNablaB = (nabla_b zip deltaNabla_b).map { a =>
        a._1.add(a._2)
      }
        .map(_.divi(batch.length))

      val avgNablaW = (nabla_w zip deltaNabla_w).map{ a =>
        a._1.add(a._2)
      }
        .map(_.divi(batch.length))

    }


  }
}
