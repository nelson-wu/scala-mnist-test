package me.nelsonwu


object App   {
  def main(args: Array[String]): Unit ={
    val loader = MnistLoader
    //val files = loader.MnistLoader.getMnistImageData(System.getProperty("user.dir"))

    val myNetwork = new Network(Vector(784, 100, 10))
    myNetwork.runNetwork(5, 3.0, 5)
  }
}
