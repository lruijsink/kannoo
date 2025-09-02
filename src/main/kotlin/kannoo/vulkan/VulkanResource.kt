package kannoo.vulkan

abstract class VulkanResource(
    protected val vulkan: Vulkan,
) {
    open fun destroy() {}

    init {
        vulkan.register(this)
    }
}
