import sys

if __name__ == "__main__":

    csv_file = sys.argv[1]

    f = open(csv_file, "r")

    m = [[0 for col in range(17)] for row in range(17)]

    for l in f.readlines():
        l = l.strip().replace(" ", "").split(",")
        
        row = int(l[1]) // 8
        col = int(l[2]) // 8

        m[row][col] = float(l[3])

    print(m) 

    for i in range(col):
        for j in range(row):
            print(m[i][j], ",", end="")
        print()