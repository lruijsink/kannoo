package kannoo.io

import kannoo.impl.FlattenLayer
import java.io.DataInputStream
import java.io.DataOutputStream

object FlattenLayerIO : LayerIO<FlattenLayer>(FlattenLayer::class) {
    override fun read(inputStream: DataInputStream): FlattenLayer =
        FlattenLayer(inputShape = inputStream.readShape())

    override fun write(layer: FlattenLayer, outputStream: DataOutputStream) {
        outputStream.writeShape(layer.inputShape)
    }
}
