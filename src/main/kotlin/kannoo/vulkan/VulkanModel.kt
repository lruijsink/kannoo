package kannoo.vulkan

import kannoo.impl.Logistic
import kannoo.impl.ReLU

class VulkanModel(val vulkan: Vulkan) {

    val inputBuffer = vulkan.createMatrixBuffer(rows = 2, cols = 3)
    val dense1 = VulkanDenseForward(vulkan, inputBuffer, outputSize = 4, ReLU)
    val dense2 = VulkanDenseForward(vulkan, dense1.output, outputSize = 4, Logistic)
    val outputBuffer = dense2.output

    val execution = vulkan.createExecution(
        dense1.commandBuffer,
        dense2.commandBuffer,
    )

    fun execute() {
        execution.submit()
    }
}
