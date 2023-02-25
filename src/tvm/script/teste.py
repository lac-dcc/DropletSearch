from itertools import combinations, permutations

#comb = combinations(["x0", "x1", "y0", "y1", "k0", "k1"], 6)
#perm = permutations(["x0", "x1", "y0", "y1", "k0", "k1"], 6)

def all_order(list_elements):
    perm = permutations(list_elements, len(list_elements))
    for i, p in enumerate(perm):
        if i == 0:
            print("if cfg[\"order\"].val == %d:" %(i))
        elif i > 0:
            print("elif cfg[\"order\"].val == %d:" %(i))
        print("    s[C].reorder(%s" %(p[0]), end="")
        for p1 in p[1:]:
            print(", %s" %(p1), end="")
        print(")")

#["", "x0", "x1", "y0", "y1", "k0", "k1", "x0y0", "x0k0", "x0x1", "x0y1", "x0k1", "x1y0", "x1k0", "x1y1", "x1k1", "y0k0", "y0y1", "y1k0", "y1k1", "k0k1", "x0x0k0"]

def all_vec(list_elements):
    print("cfg.define_knob(\"vec\", [\"\",",end="")
    for i in range(1,len(list_elements)+1):
        comb = combinations(list_elements, i)
        for iter in comb:
            print("\"", end="")
            for c in iter:
                print(c, end="")
            print("\"", end=", ")
    print("])")

all_vec(["x0", "x1", "y0", "y1", "k0", "k1"])

#all_order(["x0", "x1", "y0", "y1", "k0", "k1"])
