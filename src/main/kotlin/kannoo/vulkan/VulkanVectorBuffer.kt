package kannoo.vulkan

import kannoo.math.Vector

class VulkanVectorBuffer(
    val size: Int,
    val buffer: VulkanBuffer,
) {
    fun get(vector: Vector) {
        if (size != vector.size)
            throw IllegalArgumentException("Incompatible vector size")

        buffer.mapped.clear()
        buffer.mapped.asFloatBuffer().get(vector.elements)
        buffer.mapped.flip()
    }

    fun get(): Vector {
        val res = Vector(size)
        get(res)
        return res
    }

    fun set(vector: Vector) {
        if (size != vector.size)
            throw IllegalArgumentException("Incompatible vector size")

        buffer.mapped.clear()
        buffer.mapped.asFloatBuffer().put(vector.elements)
        buffer.mapped.flip()
    }
}

fun Vulkan.createVectorBuffer(size: Int): VulkanVectorBuffer =
    VulkanVectorBuffer(size, createBuffer(size * Float.SIZE_BYTES.toLong()))

fun Vulkan.createVectorBuffer(vector: Vector): VulkanVectorBuffer {
    val buffer = createVectorBuffer(vector.size)
    buffer.set(vector)
    return buffer
}
