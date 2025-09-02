package kannoo.vulkan

class VulkanPushConstants(
    val values: List<Int>, // TODO: Add support for other types
) {
    constructor(vararg values: Int) : this(values.toList())

    val size = values.size * Int.SIZE_BYTES
}
