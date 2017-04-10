package me.nelsonwu

import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Created by t-yowu on 4/5/2017.
  */
object MnistLoader {
  import java.io.{BufferedInputStream, FileInputStream}
  import java.util.zip.GZIPInputStream

  import org.nd4j.linalg.factory.Nd4j


  /**
    * Created by clvcooke on 6/6/2016.
    */
  object MnistLoader {

    private def gzipInputStream(s: String) = new GZIPInputStream(new BufferedInputStream(new FileInputStream(s)))

    private def read32BitInt(i: GZIPInputStream) = i.read() * 16777216 /*2^24*/ + i.read() * 65536 /*2&16*/ + i.read() * 256 /*2^8*/ + i.read()


    /**
      *
      * @param baseDirectory the directory for the standard mnist images, file names are assumed
      */
    def getMnistImageData(baseDirectory: String): (IndexedSeq[Int], IndexedSeq[Int], IndexedSeq[INDArray], IndexedSeq[INDArray]) = {
      val testLabels = readLabels(s"$baseDirectory/t10k-labels-idx1-ubyte.gz")
      val trainingLabels = readLabels(s"$baseDirectory/train-labels-idx1-ubyte.gz")
      val testImages = readImages(s"$baseDirectory/t10k-images-idx3-ubyte.gz")
      val trainingImages = readImages(s"$baseDirectory/train-images-idx3-ubyte.gz")
      (testLabels, trainingLabels, testImages.map(_.div(255)), trainingImages.map(_.div(255)))
    }

    /**
      *
      * @param filepath the full file path the labels file
      * @return
      */
    def readLabels(filepath: String) = {
      val g = gzipInputStream(filepath)
      val magicNumber = read32BitInt(g) //currently not used for anything, as assumptions are made
      val numberOfLabels = read32BitInt(g)
      1.to(numberOfLabels).map(_ => g.read())
    }

    /**
      *
      * @param filepath the full file path of the images file
      * @return
      */
    def readImages(filepath: String) = {
      val g = gzipInputStream(filepath)
      val magicNumber = read32BitInt(g) //currently not used for anything, as assumptions are made
      val numberOfImages = read32BitInt(g)
      val imageSize = read32BitInt(g) * read32BitInt(g) //cols * rows
      (1 to numberOfImages).map(_ => Nd4j.create((1 to imageSize).map(_ => g.read().toFloat).toArray))
    }

  }

}
