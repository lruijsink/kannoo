package kannoo.vulkan

import org.lwjgl.system.MemoryStack.stackPush
import org.lwjgl.system.MemoryUtil.memFree
import org.lwjgl.vulkan.VK10.VK_NULL_HANDLE
import org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT
import org.lwjgl.vulkan.VK10.vkCreateComputePipelines
import org.lwjgl.vulkan.VK10.vkCreatePipelineLayout
import org.lwjgl.vulkan.VK10.vkCreateShaderModule
import org.lwjgl.vulkan.VK10.vkDestroyPipeline
import org.lwjgl.vulkan.VK10.vkDestroyPipelineLayout
import org.lwjgl.vulkan.VK10.vkDestroyShaderModule
import org.lwjgl.vulkan.VkComputePipelineCreateInfo
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo
import org.lwjgl.vulkan.VkPushConstantRange
import org.lwjgl.vulkan.VkShaderModuleCreateInfo

class VulkanPipeline(
    vulkan: Vulkan,
    val descriptorSet: VulkanDescriptorSet,
    val pushConstants: VulkanPushConstants,
    val shader: VulkanShader,
) : VulkanResource(vulkan) {

    val shaderModule: Long = createComputeShaderModule()
    val layout: Long = createPipelineLayout()
    val handle: Long = createComputePipeline()

    private fun createComputeShaderModule(): Long = stackPush().use { stack ->
        val createInfo = VkShaderModuleCreateInfo.calloc()
            .`sType$Default`()
            .pCode(shader.code)

        val pComputeShaderModule = stack.mallocLong(1)
        vkCreateShaderModule(vulkan.device, createInfo, null, pComputeShaderModule).orThrow()
        return pComputeShaderModule.get()
    }

    private fun createPipelineLayout(): Long = stackPush().use { stack ->
        val pushConstantRange = VkPushConstantRange.calloc(1)
            .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT)
            .offset(0)
            .size(pushConstants.size)

        val pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo.calloc()
            .`sType$Default`()
            .setLayoutCount(1)
            .pSetLayouts(stack.longs(descriptorSet.layout))
            .pPushConstantRanges(pushConstantRange)

        val pPipelineLayout = stack.mallocLong(1)
        vkCreatePipelineLayout(vulkan.device, pipelineLayoutCreateInfo, null, pPipelineLayout).orThrow()
        return pPipelineLayout.get()
    }

    private fun createComputePipeline(): Long = stackPush().use { stack ->
        val shaderStageCreateInto = VkPipelineShaderStageCreateInfo.calloc()
            .`sType$Default`()
            .stage(VK_SHADER_STAGE_COMPUTE_BIT)
            .module(shaderModule)
            .pName(stack.UTF8("main"))

        val pipelineCreateInfo = VkComputePipelineCreateInfo.calloc(1)
            .`sType$Default`()
            .stage(shaderStageCreateInto)
            .layout(layout)

        val pPipeline = stack.mallocLong(1)
        vkCreateComputePipelines(vulkan.device, VK_NULL_HANDLE, pipelineCreateInfo, null, pPipeline).orThrow()
        return pPipeline.get()
    }

    override fun destroy() {
        memFree(shader.code)
        vkDestroyShaderModule(vulkan.device, shaderModule, null)
        vkDestroyPipelineLayout(vulkan.device, layout, null)
        vkDestroyPipeline(vulkan.device, handle, null)
    }
}

fun Vulkan.createPipeline(
    descriptorSet: VulkanDescriptorSet,
    pushConstants: VulkanPushConstants,
    shader: VulkanShader,
) = VulkanPipeline(this, descriptorSet, pushConstants, shader)
