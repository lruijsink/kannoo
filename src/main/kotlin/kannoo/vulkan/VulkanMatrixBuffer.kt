package kannoo.vulkan

class VulkanMatrixBuffer(
    val rows: Int,
    val cols: Int,
    val buffer: VulkanBuffer,
)

fun Vulkan.createMatrixBuffer(rows: Int, cols: Int): VulkanMatrixBuffer =
    VulkanMatrixBuffer(rows, cols, createBuffer(rows * cols * Float.SIZE_BYTES.toLong()))
