package kannoo.io

import kannoo.impl.GrayscaleConvolutionLayer
import java.io.DataInputStream
import java.io.DataOutputStream

object GrayscaleConvolutionLayerIO : LayerIO<GrayscaleConvolutionLayer>(GrayscaleConvolutionLayer::class) {
    override fun write(layer: GrayscaleConvolutionLayer, outputStream: DataOutputStream) {
        outputStream.writeTerminatedString(layer.activationFunction.serialize())
        outputStream.writeDimensions(layer.inputDimensions)
        outputStream.writeInt(layer.outputChannels)
        outputStream.writeDimensions(layer.kernelDimensions)
        outputStream.writeNTensor3(layer.kernels)
        outputStream.writeVector(layer.bias)
        outputStream.writeNullable(layer.padding) { writePadding(it) }
        outputStream.writeNullable(layer.stride) { writeDimensions(it) }
    }

    override fun read(inputStream: DataInputStream): GrayscaleConvolutionLayer {
        val activationFunction = deserializeActivationFunction(inputStream.readTerminatedString())
        val inputDimensions = inputStream.readDimensions()
        val outputChannels = inputStream.readInt()
        val kernelDimensions = inputStream.readDimensions()
        val kernels = inputStream.readNTensor3(outputChannels, kernelDimensions.height, kernelDimensions.width)
        val bias = inputStream.readVector(outputChannels)
        val padding = inputStream.readNullable { readPadding() }
        val stride = inputStream.readNullable { readDimensions() }
        return GrayscaleConvolutionLayer(
            inputDimensions = inputDimensions,
            kernels = kernels,
            bias = bias,
            padding = padding,
            stride = stride,
            activationFunction = activationFunction,
        )
    }
}
