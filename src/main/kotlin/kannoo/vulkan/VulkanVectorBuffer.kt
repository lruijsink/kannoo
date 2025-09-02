package kannoo.vulkan

class VulkanVectorBuffer(
    val size: Int,
    val buffer: VulkanBuffer,
)

fun Vulkan.createVectorBuffer(size: Int): VulkanVectorBuffer =
    VulkanVectorBuffer(size, createBuffer(size * Float.SIZE_BYTES.toLong()))
