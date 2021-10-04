using Test
using BoostedCDE

@test BoostedCDE.tri_col_ranges(3) == [1:1, 2:3, 4:6]
@test BoostedCDE.tri_n_el_to_d.([3,6,10,15]) == [2,3,4,5]
@test BoostedCDE.vecvec_triangular_view([1,2,3]) == [[1], [2,3]]
@test BoostedCDE.triangular_from_vecvec([[1], [2,3]]) == UpperTriangular([1 2; 0 3])

a = [1,2,3]
b = BoostedCDE.vecvec_triangular_view(a)
a .+= 10
@test b[1] == [11]
@test b[2] == [12, 13]


