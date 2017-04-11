package me.nelsonwu

import org.nd4j.linalg.api.buffer.util.DataTypeUtil


object App   {
  def main(args: Array[String]): Unit ={
    val loader = MnistLoader
    //val files = loader.MnistLoader.getMnistImageData(System.getProperty("user.dir"))
    //val result = test(IndexedSeq.fill(10){true})

    DataTypeUtil.setDTypeForContext("double")
    val myNetwork = new Network(Vector(784, 30, 10))
    myNetwork.runNetwork(10, 10000, 3.0, 50)
  }

  def test(results: IndexedSeq[Boolean]):Int = {
    val correctCount = results.foldLeft(0) {
      case (count, true) => count + 1
      case (count, false) => count
    }
    correctCount
  }

}
