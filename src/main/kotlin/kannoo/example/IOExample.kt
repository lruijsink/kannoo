package kannoo.example

import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.impl.DenseLayer
import kannoo.impl.ReLU
import kannoo.impl.Softmax
import kannoo.io.readModelFromFile
import kannoo.io.writeModelToFile

fun ioExample() {
    val model = Model(
        InputLayer(1),
        DenseLayer(100, ReLU),
        DenseLayer(100, Softmax),
    )
    writeModelToFile(model, "./data/example.kannoo")
    val fromDisk = readModelFromFile("./data/example.kannoo")
}
