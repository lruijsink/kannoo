package kannoo.vulkan

import org.lwjgl.system.MemoryStack.stackPush
import org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY
import org.lwjgl.vulkan.VK10.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
import org.lwjgl.vulkan.VK10.VK_PIPELINE_BIND_POINT_COMPUTE
import org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT
import org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers
import org.lwjgl.vulkan.VK10.vkBeginCommandBuffer
import org.lwjgl.vulkan.VK10.vkCmdBindDescriptorSets
import org.lwjgl.vulkan.VK10.vkCmdBindPipeline
import org.lwjgl.vulkan.VK10.vkCmdDispatch
import org.lwjgl.vulkan.VK10.vkCmdPushConstants
import org.lwjgl.vulkan.VK10.vkCreateCommandPool
import org.lwjgl.vulkan.VK10.vkDestroyCommandPool
import org.lwjgl.vulkan.VK10.vkEndCommandBuffer
import org.lwjgl.vulkan.VK10.vkResetCommandBuffer
import org.lwjgl.vulkan.VkCommandBuffer
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo
import org.lwjgl.vulkan.VkCommandBufferBeginInfo
import org.lwjgl.vulkan.VkCommandPoolCreateInfo
import java.nio.IntBuffer

class VulkanCommandBuffer(
    vulkan: Vulkan,
    val pipeline: VulkanPipeline,
    val groupCountX: Int,
    val groupCountY: Int = 1,
    val groupCountZ: Int = 1,
) : VulkanResource(vulkan) {

    val pool: Long = createCommandPool()
    val handle: VkCommandBuffer = createCommandBuffer()

    init {
        if (pipeline.pushConstantCount == 0) {
            record()
        }
    }

    private fun createCommandPool(): Long = stackPush().use { stack ->
        val commandPoolCreateInfo = VkCommandPoolCreateInfo.calloc()
            .`sType$Default`()
            .flags(if (pipeline.pushConstantCount > 0) VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT else 0)
            .queueFamilyIndex(vulkan.queueFamilyIndex)

        val pCommandPool = stack.mallocLong(1)
        vkCreateCommandPool(vulkan.device, commandPoolCreateInfo, null, pCommandPool).orThrow()
        return pCommandPool.get()
    }

    private fun createCommandBuffer(): VkCommandBuffer = stackPush().use { stack ->
        val commandBufferAllocateInfo = VkCommandBufferAllocateInfo.calloc()
            .`sType$Default`()
            .commandPool(pool)
            .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
            .commandBufferCount(1)

        val pCommandBuffer = stack.mallocPointer(1)
        vkAllocateCommandBuffers(vulkan.device, commandBufferAllocateInfo, pCommandBuffer).orThrow()
        return VkCommandBuffer(pCommandBuffer.get(), vulkan.device)
    }

    fun record(pushConstants: VulkanPushConstants? = null): VulkanCommandBuffer = stackPush().use { stack ->
        if (pushConstants != null)
            vkResetCommandBuffer(handle, 0)

        val beginInfo = VkCommandBufferBeginInfo.calloc()
            .`sType$Default`()

        vkBeginCommandBuffer(handle, beginInfo).orThrow()

        if (pushConstants != null) {
            vkCmdPushConstants(
                handle,
                pipeline.layout,
                VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                pushConstants.record(stack),
            )
        } else if (pipeline.pushConstantCount > 0) {
            throw IllegalStateException("Expected ${pipeline.pushConstantCount} but no producer was supplied")
        }

        vkCmdBindPipeline(handle, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle)

        vkCmdBindDescriptorSets(
            handle,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout,
            0,
            stack.longs(pipeline.descriptorSet.handle),
            null as IntBuffer?,
        )

        vkCmdDispatch(handle, groupCountX, groupCountY, groupCountZ)

        vkEndCommandBuffer(handle).orThrow()

        return this
    }

    override fun destroy() {
        vkDestroyCommandPool(vulkan.device, pool, null)
    }
}

fun Vulkan.createCommandBuffer(
    pipeline: VulkanPipeline,
    groupCountX: Int,
    groupCountY: Int = 1,
    groupCountZ: Int = 1,
): VulkanCommandBuffer =
    VulkanCommandBuffer(this, pipeline, groupCountX, groupCountY, groupCountZ)
