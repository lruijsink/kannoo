package kannoo.vulkan

import org.lwjgl.system.MemoryUtil.memFree

class VulkanShader(
    vulkan: Vulkan,
    fileName: String,
    val workgroupSize: Int,
) : VulkanResource(vulkan) {
    val code = readFileToNative(fileName)

    override fun destroy() {
        memFree(code)
    }
}

fun Vulkan.createShader(fileName: String, workgroupSize: Int): VulkanShader =
    VulkanShader(this, fileName, workgroupSize)
