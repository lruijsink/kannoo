package kannoo.core

import kannoo.math.Tensor
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class GradientComputer(
    model: Model,
    cost: CostFunction,
) {
    private val threadCount = Runtime.getRuntime().availableProcessors()
    private val backPropagators = List(threadCount) { BackPropagator(model, cost) }
    private val gradientAccumulators = List(threadCount) { GradientAccumulator(model) }
    private val combinedAccumulator = GradientAccumulator(model)
    private val combinedAccumulatorLock = ReentrantLock()
    private val threadPool = Executors.newFixedThreadPool(threadCount)

    fun computeGradients(samples: List<Sample>): Map<Tensor, Tensor> {
        combinedAccumulator.reset()

        val q = AtomicInteger(0)
        val futures = List(threadCount) { t ->
            threadPool.submit {
                gradientAccumulators[t].reset()
                while (true) {
                    val i = q.getAndIncrement()
                    if (i >= samples.size) break
                    backPropagators[t].calculatePartials(samples[i], gradientAccumulators[t])
                }
                combinedAccumulatorLock.withLock {
                    gradientAccumulators[t].gradients.forEach { (param, gradient) ->
                        combinedAccumulator(param, gradient)
                    }
                }
            }
        }
        futures.forEach { it.get() }

        for ((_, gradient) in combinedAccumulator.gradients)
            gradient.mapAssign { x -> if (x.isNaN() || x.isInfinite()) 0.0f else x }

        return combinedAccumulator.gradients
    }
}
