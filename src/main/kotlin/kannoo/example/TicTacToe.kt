package kannoo.example

import kannoo.core.InputLayer
import kannoo.core.Model
import kannoo.core.Sample
import kannoo.example.Eval.Draw
import kannoo.example.Eval.OWin
import kannoo.example.Eval.XWin
import kannoo.example.Square.Empty
import kannoo.example.Square.O
import kannoo.example.Square.X
import kannoo.impl.DenseLayer
import kannoo.impl.Logistic
import kannoo.impl.MeanSquaredError
import kannoo.impl.MiniBatchSGD
import kannoo.impl.ReLU
import kannoo.math.Vector

enum class Square { X, O, Empty }

val startingPlayer = X
fun winner(square: Square) = if (square == X) XWin else OWin

val Square.opponent
    get() =
        when (this) {
            X -> O
            O -> X
            Empty -> throw IllegalArgumentException("Cannot get opponent of Empty")
        }

enum class Eval { XWin, OWin, Draw }

val Eval.opposite
    get() =
        when (this) {
            XWin -> OWin
            OWin -> XWin
            Draw -> Draw
        }

typealias Board = Array<Array<Square>>

val emptyBoard = Array(3) { Array(3) { Empty } }

val Board.emptySquares get() = this.sumOf { row -> row.count { it == Empty } }

fun Board.forEachSquare(fn: (Int, Int, Square) -> Unit) {
    for (i in 0 until 3)
        for (j in 0 until 3)
            fn(i, j, this[i][j])
}

fun Board.forEachEmptySquare(fn: (Int, Int) -> Unit) =
    forEachSquare { i, j, q -> if (q == Empty) fn(i, j) }

fun Board.copy(): Board =
    Array(3) { this[it].copyOf() }

fun Board.move(i: Int, j: Int, move: Square): Board {
    val res = copy()
    res[i][j] = move
    return res
}

fun Board.eval(): Eval? {

    // Rows:
    for (i in 0 until 3)
        if (this[i][0] != Empty && this[i].all { it == this[i][0] })
            return winner(this[i][0])

    // Columns:
    for (i in 0 until 3)
        if (this[0][i] != Empty && all { row -> row[i] == this[0][i] })
            return winner(this[0][i])

    // Diagonals:
    if (this[0][0] != Empty && this[0][0] == this[1][1] && this[1][1] == this[2][2])
        return winner(this[0][0])

    if (this[0][2] != Empty && this[0][2] == this[1][1] && this[1][1] == this[2][0])
        return winner(this[0][2])

    if (this.none { Empty in it })
        return Draw

    return null
}

val cache = mutableMapOf<Board, Eval>()
val bestMoves = mutableMapOf<Board, List<Pair<Int, Int>>>()

fun solve(board: Board, player: Square): Eval {
    if (board in cache) return cache[board]!!

    val eval = board.eval()
    if (eval != null) {
        cache[board] = eval
        return eval
    }

    val playerWin = winner(player)
    val winningMoves = mutableListOf<Pair<Int, Int>>()
    val drawingMoves = mutableListOf<Pair<Int, Int>>()
    val losingMoves = mutableListOf<Pair<Int, Int>>()

    board.forEachEmptySquare { i, j ->
        val moveEval = solve(board.move(i, j, player), player.opponent)
        when (moveEval) {
            playerWin -> winningMoves += Pair(i, j)
            Draw -> drawingMoves += Pair(i, j)
            else -> losingMoves += Pair(i, j)
        }
    }

    val bestPlayEval = when {
        winningMoves.isNotEmpty() -> playerWin
        drawingMoves.isNotEmpty() -> Draw
        else -> playerWin.opposite
    }
    cache[board] = bestPlayEval

    bestMoves[board] = winningMoves.ifEmpty { drawingMoves }.ifEmpty { losingMoves }

    return bestPlayEval
}

fun Square.toInput(): Double =
    when (this) {
        X -> 1.0
        O -> -1.0
        Empty -> 0.0
    }

fun Board.toInput(): Vector =
    Vector(this.flatMap { row -> row.map { it.toInput() } }.toDoubleArray())

fun List<Pair<Int, Int>>.toTarget(): Vector =
    Vector(
        (0 until 3).flatMap { i ->
            (0 until 3).map { j ->
                if (Pair(i, j) in this) 1.0 else 0.0
            }
        }.toDoubleArray()
    )

fun Vector.toMoves(): List<Pair<Int, Int>> =
    listOfNotNull(
        if (this[0] > 0.5) Pair(0, 0) else null,
        if (this[1] > 0.5) Pair(0, 1) else null,
        if (this[2] > 0.5) Pair(0, 2) else null,
        if (this[3] > 0.5) Pair(1, 0) else null,
        if (this[4] > 0.5) Pair(1, 1) else null,
        if (this[5] > 0.5) Pair(1, 2) else null,
        if (this[6] > 0.5) Pair(2, 0) else null,
        if (this[7] > 0.5) Pair(2, 1) else null,
        if (this[8] > 0.5) Pair(2, 2) else null,
    )

fun ticTacToeExample() {
    solve(emptyBoard, startingPlayer)

    fun draw(boards: List<Board>) {
        for (i in 0 until 3) {
            boards.forEach { board -> print(board[i].joinToString(" ") { if (it == Empty) "." else it.name } + "   ") }
            print("\n")
        }
    }

    fun draw(moveLists: List<List<Pair<Int, Int>>>) {
        for (i in 0 until 3) {
            for (b in 0 until moveLists.size) {
                for (j in 0 until 3) {
                    print(if (Pair(i, j) in moveLists[b]) "# " else ". ")
                }
                print("  ")
            }
            print("\n")
        }
    }

    val trainingData: List<Sample> = bestMoves.map { (board, bestMoves) ->
        Sample(input = board.toInput(), target = bestMoves.toTarget())
    }

    val cost = MeanSquaredError
    val model = Model(
        InputLayer(3 * 3),
        DenseLayer(3 * 3 * 20, ReLU),
        DenseLayer(3 * 3, Logistic),
    )
    val sgd = MiniBatchSGD(model, cost, 25, 0.1)

    (1..100).forEach { n ->
        println()
        println("------------------------------------------------------------------------------------------------")
        println("Training round $n")
        println()
        sgd.apply(trainingData)

        println("Sparse examples:")
        val sparse = bestMoves.keys.filter { it.emptySquares > 4 }.shuffled().take(20)
        draw(sparse)
        println()
        draw(sparse.map { model.compute(it.toInput()).toMoves() })
        println()

        println("Dense examples:")
        val dense = bestMoves.keys.filter { it.emptySquares <= 4 }.shuffled().take(20)
        draw(dense)
        println()
        draw(dense.map { model.compute(it.toInput()).toMoves() })
        println()

        val costSum = trainingData.sumOf { (input, target) ->
            cost.compute(target, model.compute(input))
        }
        println("Error: " + rnd(costSum / trainingData.size.toDouble()))
        println()
    }
}
