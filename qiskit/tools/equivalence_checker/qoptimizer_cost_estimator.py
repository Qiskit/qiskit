# configurable...
# correctness_perf_ratio = 10


def histogram(matrix):
    histo = {}
    rowCount = matrix.rows
    for i in range(rowCount):
        item = str(matrix[i,0])
        if item not in histo:
            histo[item] = 1
        else:
            histo[item] = histo[item] + 1
    # print histo
    return histo

def correctness_cost_histogram(matrix1, matrix2):
    hist1 = histogram(matrix1)
    hist2 = histogram(matrix2)

    total = 0
    for key in hist1:
        val1 = hist1[key]
        if key in hist2:
            val2 = hist2[key]
            total += abs(val1-val2)
            del hist2[key]
        else:
            total += val1

    for key in hist2:
        val2 = hist2[key]
        total += val2

    hist1.clear()
    hist2.clear()

    return total



def correctness_cost_disorder(matrix1, matrix2):
    if matrix1 == matrix2:
        return 0
    else:
        return disorder_penality

def correctness_cost(cexecutor, oracle_file, current_file, disorder_penality):
    matrix = cexecutor.cache_get(oracle_file, "matrix")
    newmatrix = cexecutor.cache_get(current_file, "matrix")

    histgram_cost = correctness_cost_histogram(matrix, newmatrix)
    if histgram_cost != 0:
        return histgram_cost + disorder_penality # 100% probability for disorder penality
    else:
        return histgram_cost + correctness_cost_disorder(matrix, newmatrix) # may avoid disorder penality


def perf_cost(cexecutor, oracle_file, current_file):
    schedule = cexecutor.cache_get(oracle_file, "schedule")
    newschedule = cexecutor.cache_get(current_file, "schedule")
    _perfcost = len(newschedule) - len(schedule)
    return _perfcost






