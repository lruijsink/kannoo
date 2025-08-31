package kannoo.vulkan

import kannoo.math.Matrix
import kannoo.math.Shape
import org.lwjgl.system.MemoryStack.stackPush
import org.lwjgl.system.MemoryUtil.memFree
import org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY
import org.lwjgl.vulkan.VK10.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
import org.lwjgl.vulkan.VK10.VK_NULL_HANDLE
import org.lwjgl.vulkan.VK10.VK_PIPELINE_BIND_POINT_COMPUTE
import org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT
import org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers
import org.lwjgl.vulkan.VK10.vkAllocateDescriptorSets
import org.lwjgl.vulkan.VK10.vkBeginCommandBuffer
import org.lwjgl.vulkan.VK10.vkCmdBindDescriptorSets
import org.lwjgl.vulkan.VK10.vkCmdBindPipeline
import org.lwjgl.vulkan.VK10.vkCmdDispatch
import org.lwjgl.vulkan.VK10.vkCmdPushConstants
import org.lwjgl.vulkan.VK10.vkCreateCommandPool
import org.lwjgl.vulkan.VK10.vkCreateComputePipelines
import org.lwjgl.vulkan.VK10.vkCreateDescriptorPool
import org.lwjgl.vulkan.VK10.vkCreateDescriptorSetLayout
import org.lwjgl.vulkan.VK10.vkCreateFence
import org.lwjgl.vulkan.VK10.vkCreatePipelineLayout
import org.lwjgl.vulkan.VK10.vkCreateShaderModule
import org.lwjgl.vulkan.VK10.vkDestroyCommandPool
import org.lwjgl.vulkan.VK10.vkDestroyDescriptorPool
import org.lwjgl.vulkan.VK10.vkDestroyDescriptorSetLayout
import org.lwjgl.vulkan.VK10.vkDestroyFence
import org.lwjgl.vulkan.VK10.vkDestroyPipeline
import org.lwjgl.vulkan.VK10.vkDestroyPipelineLayout
import org.lwjgl.vulkan.VK10.vkDestroyShaderModule
import org.lwjgl.vulkan.VK10.vkEndCommandBuffer
import org.lwjgl.vulkan.VK10.vkQueueSubmit
import org.lwjgl.vulkan.VK10.vkResetFences
import org.lwjgl.vulkan.VK10.vkUpdateDescriptorSets
import org.lwjgl.vulkan.VK10.vkWaitForFences
import org.lwjgl.vulkan.VkCommandBuffer
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo
import org.lwjgl.vulkan.VkCommandBufferBeginInfo
import org.lwjgl.vulkan.VkCommandPoolCreateInfo
import org.lwjgl.vulkan.VkComputePipelineCreateInfo
import org.lwjgl.vulkan.VkDescriptorBufferInfo
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo
import org.lwjgl.vulkan.VkDescriptorPoolSize
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo
import org.lwjgl.vulkan.VkFenceCreateInfo
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo
import org.lwjgl.vulkan.VkPushConstantRange
import org.lwjgl.vulkan.VkShaderModuleCreateInfo
import org.lwjgl.vulkan.VkSubmitInfo
import org.lwjgl.vulkan.VkWriteDescriptorSet
import java.nio.IntBuffer

class VulkanMatrixMultiply(
    private val vulkan: Vulkan,
    private val shader: Shader,
    private val inputSize: Int,
    private val outputSize: Int,
    private val batchSize: Int,
) {
    private val inputBufferSize: Long = batchSize * inputSize * Float.SIZE_BYTES.toLong()
    private val weightsBufferSize: Long = inputSize * outputSize * Float.SIZE_BYTES.toLong()
    private val outputBufferSize: Long = batchSize * outputSize * Float.SIZE_BYTES.toLong()

    private val inputBuffer: VulkanBuffer = vulkan.createBuffer(inputBufferSize)
    private val weightsBuffer: VulkanBuffer = vulkan.createBuffer(weightsBufferSize)
    private val outputBuffer: VulkanBuffer = vulkan.createBuffer(outputBufferSize)

    private val descriptorSetLayout: Long
    private val descriptorPool: Long
    private val descriptorSet: Long
    private val computeShaderModule: Long
    private val pipelineLayout: Long
    private val pipeline: Long
    private val fence: Long
    private val commandPool: Long
    private val commandBuffer: VkCommandBuffer

    init {
        this.descriptorSetLayout = createDescriptorSetLayout()

        val (descriptorPool, descriptorSet) = createDescriptorSet()
        this.descriptorPool = descriptorPool
        this.descriptorSet = descriptorSet

        val (computeShaderModule, pipelineLayout, pipeline) = createComputePipeline()
        this.computeShaderModule = computeShaderModule
        this.pipelineLayout = pipelineLayout
        this.pipeline = pipeline

        val (commandPool, commandBuffer) = createCommandBuffer()
        this.commandPool = commandPool
        this.commandBuffer = commandBuffer

        this.fence = createFence()
    }

    private fun createDescriptorSetLayout(): Long = stackPush().use { stack ->
        val bindings = VkDescriptorSetLayoutBinding.calloc(3)

        bindings.get(0)
            .binding(0)
            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .descriptorCount(1)
            .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT)

        bindings.get(1)
            .binding(1)
            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .descriptorCount(1)
            .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT)

        bindings.get(2)
            .binding(2)
            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .descriptorCount(1)
            .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT)

        val createInfo = VkDescriptorSetLayoutCreateInfo.calloc()
            .`sType$Default`()
            .pBindings(bindings)

        val pDescriptorSetLayout = stack.mallocLong(1)
        vkCreateDescriptorSetLayout(vulkan.device, createInfo, null, pDescriptorSetLayout).orThrow()
        return pDescriptorSetLayout.get(0)
    }

    private fun createDescriptorSet(): Pair<Long, Long> = stackPush().use { stack ->
        val poolSize = VkDescriptorPoolSize.calloc(1)
            .type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .descriptorCount(3)

        val poolCreateInfo = VkDescriptorPoolCreateInfo.calloc()
            .`sType$Default`()
            .maxSets(1)
            .pPoolSizes(poolSize)

        val pDescriptorPool = stack.mallocLong(1)
        vkCreateDescriptorPool(vulkan.device, poolCreateInfo, null, pDescriptorPool).orThrow()
        val descriptorPool = pDescriptorPool.get()

        val allocateInfo = VkDescriptorSetAllocateInfo.calloc()
            .`sType$Default`()
            .descriptorPool(descriptorPool)
            .pSetLayouts(stack.longs(descriptorSetLayout))

        val pDescriptorSets = stack.mallocLong(1)
        vkAllocateDescriptorSets(vulkan.device, allocateInfo, pDescriptorSets).orThrow()
        val descriptorSet = pDescriptorSets.get(0)

        val writeDescriptorSet = VkWriteDescriptorSet.calloc(3)

        val inputBufferInfo = VkDescriptorBufferInfo.calloc(1)
            .buffer(inputBuffer.handle)
            .offset(0)
            .range(inputBufferSize)

        writeDescriptorSet.get(0)
            .`sType$Default`()
            .dstSet(pDescriptorSets.get(0))
            .dstBinding(0)
            .descriptorCount(1)
            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .pBufferInfo(inputBufferInfo)

        val weightsBufferInfo = VkDescriptorBufferInfo.calloc(1)
            .buffer(weightsBuffer.handle)
            .offset(0)
            .range(weightsBufferSize)

        writeDescriptorSet.get(1)
            .`sType$Default`()
            .dstSet(pDescriptorSets.get(0))
            .dstBinding(1)
            .descriptorCount(1)
            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .pBufferInfo(weightsBufferInfo)

        val outputBufferInfo = VkDescriptorBufferInfo.calloc(1)
            .buffer(outputBuffer.handle)
            .offset(0)
            .range(outputBufferSize)

        writeDescriptorSet.get(2)
            .`sType$Default`()
            .dstSet(pDescriptorSets.get(0))
            .dstBinding(2)
            .descriptorCount(1)
            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            .pBufferInfo(outputBufferInfo)

        vkUpdateDescriptorSets(vulkan.device, writeDescriptorSet, null)

        return Pair(descriptorPool, descriptorSet)
    }

    private fun createComputePipeline(): Triple<Long, Long, Long> = stackPush().use { stack ->
        val createInfo = VkShaderModuleCreateInfo.calloc()
            .`sType$Default`()
            .pCode(shader.code)

        val pComputeShaderModule = stack.mallocLong(1)
        vkCreateShaderModule(vulkan.device, createInfo, null, pComputeShaderModule).orThrow()
        val computeShaderModule = pComputeShaderModule.get(0)

        val pushConstantRange = VkPushConstantRange.calloc(1)
            .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT)
            .offset(0)
            .size(3 * Int.SIZE_BYTES)

        val pipelineLayoutCreateInfo = VkPipelineLayoutCreateInfo.calloc()
            .`sType$Default`()
            .setLayoutCount(1)
            .pSetLayouts(stack.longs(descriptorSetLayout))
            .pPushConstantRanges(pushConstantRange)

        val pPipelineLayout = stack.mallocLong(1)
        vkCreatePipelineLayout(vulkan.device, pipelineLayoutCreateInfo, null, pPipelineLayout).orThrow()
        val pipelineLayout = pPipelineLayout.get(0)

        val shaderStageCreateInto = VkPipelineShaderStageCreateInfo.calloc()
            .`sType$Default`()
            .stage(VK_SHADER_STAGE_COMPUTE_BIT)
            .module(computeShaderModule)
            .pName(stack.UTF8("main"))

        val pipelineCreateInfo = VkComputePipelineCreateInfo.calloc(1)
            .`sType$Default`()
            .stage(shaderStageCreateInto)
            .layout(pipelineLayout)

        val pPipeline = stack.mallocLong(1)
        vkCreateComputePipelines(vulkan.device, VK_NULL_HANDLE, pipelineCreateInfo, null, pPipeline).orThrow()
        val pipeline = pPipeline.get(0)

        return Triple(computeShaderModule, pipelineLayout, pipeline)
    }

    private fun createCommandBuffer(): Pair<Long, VkCommandBuffer> = stackPush().use { stack ->
        val commandPoolCreateInfo = VkCommandPoolCreateInfo.calloc()
            .`sType$Default`()
            .flags(0)
            .queueFamilyIndex(vulkan.queueFamilyIndex)

        val pCommandPool = stack.mallocLong(1)
        vkCreateCommandPool(vulkan.device, commandPoolCreateInfo, null, pCommandPool).orThrow()
        val commandPool = pCommandPool.get(0)

        val commandBufferAllocateInfo = VkCommandBufferAllocateInfo.calloc()
            .`sType$Default`()
            .commandPool(commandPool)
            .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
            .commandBufferCount(1)

        val pCommandBuffer = stack.mallocPointer(1)
        vkAllocateCommandBuffers(vulkan.device, commandBufferAllocateInfo, pCommandBuffer).orThrow()
        val commandBuffer = VkCommandBuffer(pCommandBuffer.get(0), vulkan.device)

        val beginInfo = VkCommandBufferBeginInfo.calloc()
            .`sType$Default`()

        vkBeginCommandBuffer(commandBuffer, beginInfo).orThrow()

        setPushConstants(commandBuffer)

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout,
            0,
            stack.longs(descriptorSet),
            null as IntBuffer?,
        )
        vkCmdDispatch(
            commandBuffer,
            (outputSize + shader.workgroupSize - 1) / shader.workgroupSize,
            (batchSize + shader.workgroupSize - 1) / shader.workgroupSize,
            1,
        )
        vkEndCommandBuffer(commandBuffer).orThrow()
        return Pair(commandPool, commandBuffer)
    }

    private fun setPushConstants(commandBuffer: VkCommandBuffer) = stackPush().use { stack ->
        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            stack.ints(inputSize, outputSize, batchSize),
        )
    }

    private fun createFence(): Long = stackPush().use { stack ->
        val fenceCreateInfo = VkFenceCreateInfo.calloc()
            .`sType$Default`()
            .flags(0)

        val pFence = stack.mallocLong(1)
        vkCreateFence(vulkan.device, fenceCreateInfo, null, pFence).orThrow()
        return pFence.get(0)
    }

    fun multiplyTransposed(left: Matrix, right: Matrix, destination: Matrix) {
        if (left.shape != Shape(batchSize, inputSize))
            throw IllegalArgumentException("Expecting $batchSize x $inputSize left argument, got ${left.shape}")

        if (right.shape != Shape(outputSize, inputSize))
            throw IllegalArgumentException("Expecting $outputSize x $inputSize right argument, got ${right.shape}")

        if (destination.shape != Shape(batchSize, outputSize))
            throw IllegalArgumentException("Expecting $batchSize x $outputSize destination, got ${destination.shape}")

        setInput(left)
        setWeights(right)
        runCommandBuffer()
        getOutput(destination)
    }

    fun multiplyTransposed(left: Matrix, right: Matrix): Matrix {
        val destination = Matrix(batchSize, outputSize)
        multiplyTransposed(left, right, destination)
        return destination
    }

    private fun setWeights(weights: Matrix): Unit = stackPush().use { stack ->
        val buffer = weightsBuffer.mapped.asFloatBuffer()
        weights.slices.forEach { buffer.put(it.elements) }
        buffer.flip()
    }

    private fun setInput(batch: Matrix): Unit = stackPush().use { stack ->
        val buffer = inputBuffer.mapped.asFloatBuffer()
        batch.slices.forEach { buffer.put(it.elements) }
        buffer.flip()
    }

    private fun runCommandBuffer(): Unit = stackPush().use { stack ->
        val submitInfo = VkSubmitInfo.calloc(stack)
            .`sType$Default`()
            .pCommandBuffers(stack.pointers(commandBuffer))

        vkResetFences(vulkan.device, fence)
        vkQueueSubmit(vulkan.queue, submitInfo, fence).orThrow()
        vkWaitForFences(vulkan.device, fence, true, 100000000000).orThrow()
    }

    private fun getOutput(destination: Matrix): Unit = stackPush().use { stack ->
        val buffer = outputBuffer.mapped.asFloatBuffer()
        for (i in 0 until batchSize) buffer.get(destination.slices[i].elements)
        buffer.flip()
    }

    fun destroy() {
        memFree(shader.code)
        inputBuffer.destroy()
        weightsBuffer.destroy()
        outputBuffer.destroy()
        vkDestroyFence(vulkan.device, fence, null)
        vkDestroyCommandPool(vulkan.device, commandPool, null)
        vkDestroyShaderModule(vulkan.device, computeShaderModule, null)
        vkDestroyDescriptorPool(vulkan.device, descriptorPool, null)
        vkDestroyDescriptorSetLayout(vulkan.device, descriptorSetLayout, null)
        vkDestroyPipelineLayout(vulkan.device, pipelineLayout, null)
        vkDestroyPipeline(vulkan.device, pipeline, null)
    }
}
