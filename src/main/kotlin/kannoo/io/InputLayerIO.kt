package kannoo.io

import kannoo.core.InputLayer
import java.io.DataInputStream
import java.io.DataOutputStream

fun DataInputStream.readInputLayer(): InputLayer =
    InputLayer(readShape())

fun DataOutputStream.writeInputLayer(inputLayer: InputLayer) {
    writeShape(inputLayer.shape)
}
