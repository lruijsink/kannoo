package kannoo.impl

import kannoo.core.GradientReceiver
import kannoo.math.Dimensions
import kannoo.math.Matrix
import kannoo.math.NTensor
import kannoo.math.Padding
import kannoo.math.Tensor
import kannoo.math.Vector
import kannoo.math.ZeroPadding
import kannoo.math.matrix
import kannoo.math.randomMatrix
import kannoo.math.tensor
import kannoo.math.vector
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import org.mockito.kotlin.argumentCaptor
import org.mockito.kotlin.mock
import org.mockito.kotlin.refEq
import org.mockito.kotlin.verify
import org.mockito.kotlin.verifyNoMoreInteractions

class GrayscaleConvolutionLayerTest {

    private val gradientReceiverMock = mock<GradientReceiver>()
    private val gradientCaptor = argumentCaptor<Tensor>()

    @Test
    fun `Delta input without padding or stride`() {
        val X = randomMatrix(3, 3) // doesn't actually matter
        val dY = tensor(
            matrix(
                vector(1f, 2f),
                vector(3f, 4f),
            ),
        )
        val K = tensor(
            matrix(
                vector(1f, 0f),
                vector(-1f, 2f),
            ),
        )
        val layer = GrayscaleConvolutionLayer(
            inputDimensions = X.dimensions,
            kernels = K,
            bias = Vector(K.size),
            activationFunction = ReLU,
        )
        assertEquals(
            matrix(
                vector(2f, 3f, -2f),
                vector(6f, 5f, -2f),
                vector(0f, 3f, 4f),
            ),
            layer.deltaInput(dY, X),
        )
    }

    @Test
    fun `No padding or stride`() {
        val input = matrix(
            vector(1f, 2f, 0f, 1f),
            vector(3f, 1f, 2f, 2f),
            vector(0f, 1f, 3f, 1f),
            vector(2f, 0f, 1f, 3f),
        )
        val kernels = tensor(
            matrix(
                vector(1f, -1f),
                vector(0f, 1f),
            ),
        )
        val bias = vector(0.1f)
        val layer = GrayscaleConvolutionLayer(
            inputDimensions = input.dimensions,
            kernels = kernels,
            bias = bias,
            activationFunction = ReLU,
        )
        assertEquals(
            tensor(
                matrix(
                    vector(0.1f, 4.1f, 1.1f),
                    vector(3.1f, 2.1f, 1.1f),
                    vector(-0.9f, -0.9f, 5.1f),
                ),
            ),
            layer.preActivation(input),
        )
    }

    @Test
    fun `Multiple output channels`() {
        val input = matrix(
            vector(1f, 2f, 0f, 1f),
            vector(3f, 1f, 2f, 2f),
            vector(0f, 1f, 3f, 1f),
            vector(2f, 0f, 1f, 3f),
        )
        val kernels = tensor(
            matrix(
                vector(1f, -1f),
                vector(0f, 1f),
            ),
            matrix(
                vector(2f, 0f),
                vector(1f, -3f),
            ),
        )
        val bias = vector(0.1f, 0.2f)
        val layer = GrayscaleConvolutionLayer(
            inputDimensions = input.dimensions,
            kernels = kernels,
            bias = bias,
            activationFunction = ReLU,
        )
        assertEquals(
            tensor(
                matrix(
                    vector(0.1f, 4.1f, 1.1f),
                    vector(3.1f, 2.1f, 1.1f),
                    vector(-0.9f, -0.9f, 5.1f),
                ),
                matrix(
                    vector(2.2f, -0.8f, -3.8f),
                    vector(3.2f, -5.8f, 4.2f),
                    vector(2.2f, -0.8f, -1.8f),
                ),
            ),
            layer.preActivation(input),
        )
    }

    @Test
    fun `With padding and stride`() {
        val input = matrix(
            vector(1f, 2f, 3f, 4f),
            vector(5f, 6f, 7f, 8f),
            vector(9f, 10f, 11f, 12f),
            vector(13f, 14f, 15f, 16f),
        )
        val kernel = matrix(
            vector(1f, 0f),
            vector(0f, -1f),
        )
        val kernels = tensor(kernel)
        val bias = vector(0f)
        val layer = GrayscaleConvolutionLayer(
            inputDimensions = input.dimensions,
            kernels = kernels,
            bias = bias,
            padding = Padding(1, 1, ZeroPadding),
            stride = Dimensions(2, 2),
            activationFunction = ReLU,
        )
        assertEquals(
            matrix(
                vector(-1f, -3f, 0f),
                vector(-9f, -5f, 8f),
                vector(0f, 14f, 16f),
            ),
            layer.preActivation(input)[0],
        )
        val deltaPreActivation = tensor(
            matrix(
                vector(1f, 2f, 3f),
                vector(4f, 5f, 6f),
                vector(7f, 8f, 9f),
            ),
        )
        assertEquals(
            matrix(
                vector(-1f, 0f, -2f, 0f),
                vector(0f, 5f, 0f, 6f),
                vector(-4f, 0f, -5f, 0f),
                vector(0f, 8f, 0f, 9f),
            ),
            layer.deltaInput(deltaPreActivation, input),
        )

        layer.gradients(deltaPreActivation, input, gradientReceiverMock)
        val (kernelsGradient, biasGradient) = captureGradients(kernels, bias)

        assertEquals(vector(45f), biasGradient)

        assertEquals(
            tensor(
                matrix(
                    vector(334f, 266f),
                    vector(138f, 98f),
                ),
            ),
            kernelsGradient,
        )
    }

    @Test
    fun `With non-square input, kernel, padding, and stride`() {
        val input = matrix(
            vector(1f, 2f, 3f, 4f),
            vector(5f, 6f, 7f, 8f),
            vector(9f, 10f, 11f, 12f),
        )
        val kernel = matrix(
            vector(1f, 0f, -1f),
            vector(2f, 1f, 0f),
        )
        val kernels = tensor(kernel)
        val bias = vector(0f)
        val layer = GrayscaleConvolutionLayer(
            inputDimensions = input.dimensions,
            kernels = kernels,
            bias = bias,
            padding = Padding(height = 1, width = 0, scheme = ZeroPadding),
            stride = Dimensions(height = 2, width = 1),
            activationFunction = ReLU,
        )
        assertEquals(
            matrix(
                vector(4f, 7f),
                vector(26f, 29f),
            ),
            layer.preActivation(input)[0],
        )
        val deltaPreActivation = tensor(
            matrix(
                vector(1f, 2f),
                vector(3f, 4f),
            ),
        )
        assertEquals(
            matrix(
                vector(2f, 5f, 2f, 0f),
                vector(3f, 4f, -3f, -4f),
                vector(6f, 11f, 4f, 0f),
            ),
            layer.deltaInput(deltaPreActivation, input),
        )

        layer.gradients(deltaPreActivation, input, gradientReceiverMock)
        val (kernelsGradient, biasGradient) = captureGradients(kernels, bias)

        assertEquals(vector(10f), biasGradient)

        assertEquals(
            tensor(
                matrix(
                    vector(39f, 46f, 53f),
                    vector(72f, 82f, 92f),
                ),
            ),
            kernelsGradient,
        )
    }

    private fun captureGradients(kernels: NTensor<Matrix>, bias: Vector): Pair<NTensor<Matrix>, Vector> {
        verify(gradientReceiverMock).invoke(refEq(kernels), gradientCaptor.capture())
        verify(gradientReceiverMock).invoke(refEq(bias), gradientCaptor.capture())
        verifyNoMoreInteractions(gradientReceiverMock)

        @Suppress("UNCHECKED_CAST")
        return (gradientCaptor.firstValue as NTensor<Matrix>) to (gradientCaptor.lastValue as Vector)
    }
}
