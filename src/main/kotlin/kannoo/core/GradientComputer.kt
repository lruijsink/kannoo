package kannoo.core

import kannoo.math.Tensor
import kannoo.math.TensorBase
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class GradientComputer(
    model: Model,
    cost: CostFunction,
) {
    private val threadCount = Runtime.getRuntime().availableProcessors()
    private val backPropagator = BackPropagator(model, cost)
    private val gradientReceivers = List(threadCount) { GradientReceiver(model) }
    private val combined = GradientReceiver(model)
    private val combinedLock = ReentrantLock()
    private val executorService = Executors.newFixedThreadPool(threadCount)

    fun computeGradients(samples: List<Sample<*>>): Map<TensorBase, TensorBase> {
        val q = AtomicInteger(0)
        val futures = List(threadCount) { t ->
            executorService.submit {
                gradientReceivers[t].reset()
                while (true) {
                    val i = q.getAndIncrement()
                    if (i >= samples.size) break
                    backPropagator.calculatePartials(samples[i], gradientReceivers[t])
                }
                combinedLock.withLock {
                    gradientReceivers[t].apply { param, gradient ->
                        combined(param, gradient)
                    }
                }
            }
        }
        combined.reset()
        futures.forEach { it.get() }

        for ((_, gradient) in combined.gradients)
            gradient.mapAssign { x -> if (x.isNaN() || x.isInfinite()) 0.0f else x }

        return combined.gradients
    }
}
