package kannoo.io

import kannoo.impl.ConvolutionLayer
import java.io.DataInputStream
import java.io.DataOutputStream

object ConvolutionLayerIO : LayerIO<ConvolutionLayer>(ConvolutionLayer::class) {
    override fun write(layer: ConvolutionLayer, outputStream: DataOutputStream) {
        outputStream.writeTerminatedString(layer.activationFunction.serialize())
        outputStream.writeDimensions(layer.inputDimensions)
        outputStream.writeShape(layer.kernels.shape)
        outputStream.writeTensor4(layer.kernels)
        outputStream.writeVector(layer.bias)
        outputStream.writeNullable(layer.padding) { writePadding(it) }
        outputStream.writeNullable(layer.stride) { writeDimensions(it) }
    }

    override fun read(inputStream: DataInputStream): ConvolutionLayer {
        val activationFunction = deserializeActivationFunction(inputStream.readTerminatedString())
        val inputDimensions = inputStream.readDimensions()
        val (outputChannels, inputChannels, kernelHeight, kernelWidth) = inputStream.readShape()
        val kernels = inputStream.readTensor4(outputChannels, inputChannels, kernelHeight, kernelWidth)
        val bias = inputStream.readVector(outputChannels)
        val padding = inputStream.readNullable { readPadding() }
        val stride = inputStream.readNullable { readDimensions() }
        return ConvolutionLayer(
            inputDimensions = inputDimensions,
            kernels = kernels,
            bias = bias,
            padding = padding,
            stride = stride,
            activationFunction = activationFunction,
        )
    }
}
