package kannoo.vulkan

import org.lwjgl.system.MemoryStack.stackPush
import org.lwjgl.vulkan.VK10.vkCreateFence
import org.lwjgl.vulkan.VK10.vkDestroyFence
import org.lwjgl.vulkan.VK10.vkQueueSubmit
import org.lwjgl.vulkan.VK10.vkResetFences
import org.lwjgl.vulkan.VK10.vkWaitForFences
import org.lwjgl.vulkan.VkFenceCreateInfo
import org.lwjgl.vulkan.VkSubmitInfo

class VulkanExecution(
    val vulkan: Vulkan,
    commandBuffers: List<VulkanCommandBuffer>,
) {
    constructor(vulkan: Vulkan, vararg commandBuffers: VulkanCommandBuffer) : this(vulkan, commandBuffers.toList())

    private val fence = createFence()
    private val commandBufferHandles = commandBuffers.map { it.handle }.toTypedArray()

    private fun createFence(): Long = stackPush().use { stack ->
        val fenceCreateInfo = VkFenceCreateInfo.calloc()
            .`sType$Default`()
            .flags(0)

        val pFence = stack.mallocLong(1)
        vkCreateFence(vulkan.device, fenceCreateInfo, null, pFence).orThrow()
        return pFence.get(0)
    }

    fun submit() = stackPush().use { stack ->
        val submitInfo = VkSubmitInfo.calloc(stack)
            .`sType$Default`()
            .pCommandBuffers(stack.pointers(*commandBufferHandles))

        vkResetFences(vulkan.device, fence)
        vkQueueSubmit(vulkan.queue, submitInfo, fence).orThrow()
        vkWaitForFences(vulkan.device, fence, true, 100000000000).orThrow()
    }

    fun destroy() {
        vkDestroyFence(vulkan.device, fence, null)
    }
}

fun Vulkan.createExecution(commandBuffers: List<VulkanCommandBuffer>): VulkanExecution =
    VulkanExecution(this, commandBuffers)

fun Vulkan.createExecution(vararg commandBuffers: VulkanCommandBuffer): VulkanExecution =
    VulkanExecution(this, *commandBuffers)
