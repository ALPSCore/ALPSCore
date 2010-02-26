from pyalps.hlist import HList

hl = HList([[1,2,3],[4,5]])

print hl
# [[1,2,3],[4,5]]

# !!! Testing linear access

print hl[0]
# 1

print hl[0:2]
# [1, 2]

# !!! Testing 'recursive' access

print hl[0,0]
# 1
print hl[1,1]
# 5

# !!! Linear assignment
hl[0] = 27
print hl[0]
print hl[0,0]
# 27

hl[1,1] = 13
print hl[1,1]
print hl[4]
# 13