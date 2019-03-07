import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object DL4JTest extends App {
  private val model: MultiLayerNetwork = KerasModelImport.importKerasSequentialModelAndWeights(
//    "/Users/etangrundstein/workspace/testing-dl4j/src/main/resources/model_RNN.json",
    "/Users/etangrundstein/workspace/testing-dl4j/src/main/resources/model_RNN_weights_save.h5"
  )

//  private val reader = new CSVRecordReader(0)
//  reader.initialize(new FileSplit(new ClassPathResource("temp_rnn_data_for_validation.csv").getFile))
//
//  private val batchSize = 150
//  new RecordReaderDataSetIterator(reader, batchSize)

//  model.init()


  val doubleArr: Array[Double] = Array(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2)
  private val input: INDArray = Nd4j.create(doubleArr)
  private val array: INDArray = model.output(input)
  println(array)
}
