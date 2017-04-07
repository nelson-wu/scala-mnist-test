package me.nelsonwu

object App   {
  def main(args: Array[String]): Unit ={
    val loader = MnistLoader
    val files = loader.MnistLoader.getMnistImageData(System.getProperty("user.dir"))


  }
}
