package kannoo.io

import kannoo.core.InputLayer
import java.io.DataInputStream
import java.io.DataOutputStream

fun DataInputStream.readInputLayer(): InputLayer =
    InputLayer(readInt())

fun DataOutputStream.writeInputLayer(inputLayer: InputLayer) {
    writeInt(inputLayer.size)
}
