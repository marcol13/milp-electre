def get_preference(mat, i, j):
        if mat[i][j] > mat[j][i]:
            return "P+"
        elif mat[j][i] > mat[i][j]:
            return "P-"
        elif mat[i][j] == mat[j][i] == 1:
            return "I"
        else:
            return "R"