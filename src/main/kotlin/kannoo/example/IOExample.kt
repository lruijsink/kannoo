package kannoo.example

import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.impl.ReLU
import kannoo.impl.Softmax
import kannoo.impl.denseLayer
import kannoo.io.readModelFromFile
import kannoo.io.writeModelToFile

fun ioExample() {
    val model = Model(
        InputLayer(1),
        denseLayer(100, ReLU),
        denseLayer(100, Softmax),
    )
    writeModelToFile(model, "./data/example.kannoo")
    val fromDisk = readModelFromFile("./data/example.kannoo")
}
