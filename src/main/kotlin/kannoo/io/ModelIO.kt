package kannoo.io

import kannoo.core.Model
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.InputStream
import java.io.OutputStream

fun DataInputStream.readModel(): Model {
    val inputLayer = readInputLayer()
    val innerLayerCount = readInt()
    val layers = List(innerLayerCount) { readLayer() }
    return Model(inputLayer, layers)
}

fun InputStream.readModel(): Model =
    DataInputStream(this).readModel()

fun DataOutputStream.writeModel(model: Model) {
    writeInputLayer(model.inputLayer)
    writeInt(model.layers.size)
    model.layers.forEach { writeLayer(it) }
}

fun OutputStream.writeModel(model: Model) {
    DataOutputStream(this).writeModel(model)
}
