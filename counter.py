def has_dup(A):
    n = len(A)
    if n == 2 and A[0] == A[1]:
        return True
    elif n > 2:
        print(A[:n//2], A[n//2:])
        return has_dup(A[:n//2]) or has_dup(A[n//2:])
    else:
        return False


print(has_dup([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
print(has_dup([1,1,2,3,4,5]))
print(has_dup([1,2,1,3,4,5]))
