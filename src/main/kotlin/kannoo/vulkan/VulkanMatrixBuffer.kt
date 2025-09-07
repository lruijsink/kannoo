package kannoo.vulkan

import kannoo.math.Matrix
import kannoo.math.randomMatrix

class VulkanMatrixBuffer(
    val rows: Int,
    val cols: Int,
    val buffer: VulkanBuffer,
) {
    fun get(matrix: Matrix) {
        if (matrix.rows != rows || matrix.cols != cols)
            throw IllegalArgumentException("Incompatible matrix size, expected $rows x $cols but got ${matrix.shape}")

        val buffer = buffer.mapped.asFloatBuffer()
        buffer.clear()
        for (i in 0 until rows) buffer.get(matrix.slices[i].elements)
        buffer.flip()
    }

    fun get(): Matrix {
        val res = Matrix(rows, cols)
        get(res)
        return res
    }

    fun set(matrix: Matrix) {
        if (matrix.rows != rows || matrix.cols != cols)
            throw IllegalArgumentException("Incompatible matrix size, expected $rows x $cols but got ${matrix.shape}")

        val buffer = buffer.mapped.asFloatBuffer()
        buffer.clear()
        matrix.slices.forEach { buffer.put(it.elements) }
        buffer.flip()
    }

    fun randomize() = apply {
        set(randomMatrix(rows, cols))
    }
}

fun Vulkan.createMatrixBuffer(rows: Int, cols: Int): VulkanMatrixBuffer =
    VulkanMatrixBuffer(rows, cols, createBuffer(rows * cols * Float.SIZE_BYTES.toLong()))

fun Vulkan.createMatrixBuffer(matrix: Matrix): VulkanMatrixBuffer {
    val buffer = createMatrixBuffer(matrix.rows, matrix.cols)
    buffer.set(matrix)
    return buffer
}
