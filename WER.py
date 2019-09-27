import numpy as np

def WER(r,h):
    D = np.zeros((len(r)+1, len(h)+1),  dtype=np.uint8)
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                D[0][j] = j
            elif j == 0:
                D[i][0] = i

    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                D[i][j] = D[i-1][j-1]
            else:
                sub = D[i-1][j-1]+1
                ins = D[i][j-1]+1
                delete = D[i-1][j]+1
                D[i][j] = min(sub,ins,delete)

    return (D[len(r)][len(h)]) / len(r) * 100


def LevenshteinDistance(real, hypothesis):
    ## for all i and j, d[i,j] will hold the Levenshtein distance between
    ## the first i characters of s and the first j characters of t
    ## note that d has (m+1)*(n+1) values
    r, h = real, hypothesis
    m, n = len(r), len(h)
    d = np.zeros((m+1, n+1),  dtype=np.uint8)

    ## source prefixes can be transformed into empty string by
    ## dropping all characters
    for i in range(0,m):
        d[i, 0] = i

    ## target prefixes can be reached from empty source prefix
    ##by inserting every character
    for j in range(0,n):
        d[0, j] = j

    for j in range(n):
        for i in range(m):
            if r[i] ==  h[j]:
                substitutionCost = 0
            else:
                substitutionCost = 1

                d[i, j] = min(d[i-1, j] + 1,                   ## deletion
                          d[i, j-1] + 1,                   ## insertion
                          d[i-1, j-1] + substitutionCost)  ## substitution

    return d[m, n]


if __name__ == "__main__":
    r = ["how","are","your","pets","gary"]
    h = ["who", "are", "pets", "gary"]
    #h = ["how","are","your","pets","gary"]
    print('WER', WER(r,h))

    r2 = [1,2,4,56,3]
    h2 = [1,2,4,346,3]
    print('CER', WER(r2,h2))