package kannoo.io

import kannoo.impl.DenseLayer
import java.io.DataInputStream
import java.io.DataOutputStream

object DenseLayerIO : LayerIO<DenseLayer>(DenseLayer::class) {
    override fun write(layer: DenseLayer, outputStream: DataOutputStream) {
        outputStream.writeTerminatedString(layer.activationFunction.serialize())
        outputStream.writeInt(layer.weights.rows)
        outputStream.writeInt(layer.weights.cols)
        outputStream.writeMatrix(layer.weights)
        outputStream.writeVector(layer.bias)
    }

    override fun read(inputStream: DataInputStream): DenseLayer {
        val activationFunction = deserializeActivationFunction(inputStream.readTerminatedString())
        val rows = inputStream.readInt()
        val cols = inputStream.readInt()
        val weights = inputStream.readMatrix(rows = rows, cols = cols)
        val bias = inputStream.readVector(size = rows)
        return DenseLayer(weights, bias, activationFunction)
    }
}
