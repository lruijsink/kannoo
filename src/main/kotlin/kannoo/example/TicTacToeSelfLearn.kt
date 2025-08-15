package kannoo.example

import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.impl.CrossEntropyLoss
import kannoo.impl.MeanSquaredError
import kannoo.impl.MiniBatchSGD
import kannoo.impl.ReLU
import kannoo.impl.Softmax
import kannoo.impl.denseLayer
import kannoo.math.Vector
import kannoo.math.sumOf
import kannoo.math.vector
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.random.Random

fun <T> chooseDistributed(choices: List<T>, weight: (T) -> Float): T {
    val totalWeight = choices.sumOf { weight(it) }
    val sample = Random.nextFloat() * totalWeight
    var accumulator = 0f
    for (choice in choices) {
        accumulator += weight(choice)
        if (sample < accumulator) return choice
    }
    return choices.last()
}

fun ponder(board: Board, play: Square, eval: (Board, Eval) -> Float): Board =
    chooseDistributed(moves(board, play)) { eval(it, winner(play)) }

fun moves(board: Board, play: Square): List<Board> {
    if (board.eval() != null) return listOf()
    val moves = mutableListOf<Board>()
    board.forEachEmptySquare { i, j -> moves += board.move(i, j, play) }
    return moves
}

fun Vector.scoreOf(eval: Eval): Float =
    when (eval) {
        Eval.XWin -> this[0]
        Eval.Draw -> this[1]
        Eval.OWin -> this[2]
    }

fun Eval.toTarget(): Vector =
    when (this) {
        Eval.XWin -> vector(1f, 0f, 0f)
        Eval.Draw -> vector(0f, 1f, 0f)
        Eval.OWin -> vector(0f, 0f, 1f)
    }

fun Board.toPlay(): Square {
    val xs = this.sumOf { row -> row.count { it == Square.X } }
    val os = this.sumOf { row -> row.count { it == Square.O } }
    return if (xs == os) Square.X else Square.O
}

fun inputOf(board: Board): Vector {
    val squares = board.flatMap { row ->
        row.flatMap { square ->
            listOf(
                if (square == Square.X) 1f else 0f,
                if (square == Square.O) 1f else 0f,
            )
        }
    }
    val player = if (board.toPlay() == Square.X) listOf(1f, 0f) else listOf(0f, 1f)
    return Vector((squares + player).toFloatArray())
}

fun trainingDataOf(boards: List<Board>, eval: Eval): List<Sample> =
    boards.map { board -> Sample(input = inputOf(board), target = eval.toTarget()) }

fun meanCostAgainstPerfect(model: Model, perfectPlays: List<Pair<Vector, Vector>>): Float {
    var totalCost = 0f
    val cost = MeanSquaredError
    perfectPlays.forEach { (target, input) ->
        totalCost += cost.compute(target, model.compute(input))
    }
    return totalCost / cache.size
}

fun playGame(model: Model): List<Board> {
    var board = emptyBoard
    var play = Square.X
    val moves = mutableListOf<Board>()
    while (board.eval() == null) {
        board = ponder(board, play) { next, eval ->
            model.compute(inputOf(next)).scoreOf(eval)
        }
        play = play.opponent
        moves += board
    }
    return moves
}

fun ticTacToeSelfLearn() {
    solve(emptyBoard, Square.X)
    val perfectPlays = cache.map { (board, eval) -> Pair(eval.toTarget(), inputOf(board)) }

    val model = Model(
        InputLayer(2 * 9 + 2),
        denseLayer(3 * 3 * 20, ReLU),
        denseLayer(3 * 3 * 10, ReLU),
        denseLayer(3 * 3 * 5, ReLU),
        denseLayer(3 * 3 * 2, ReLU),
        denseLayer(3, Softmax)
    )
    val sgd = MiniBatchSGD(
        model = model,
        cost = CrossEntropyLoss,
        learningRate = 0.1f,
        batchSize = 100,
    )

    repeat(100) { i ->
        println("Round ${i + 1}:")

        repeat(100) { j ->
            val trainingData = mutableListOf<Sample>()
            val tdl = ReentrantLock()
            val threads = List(20) {
                Thread {
                    val g = List(50) {
                        val moves = playGame(model)
                        trainingDataOf(moves, moves.last().eval()!!)
                    }.flatten()
                    tdl.withLock {
                        trainingData += g
                    }
                }
            }
            threads.forEach { it.start() }
            threads.forEach { it.join() }

            sgd.apply(trainingData)
        }

        printBoards(playGame(model))
        println("Mean cost w.r.t. perfect: ${meanCostAgainstPerfect(model, perfectPlays)}")
        println()
    }
}
