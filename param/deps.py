import sys
cpu_gpu = sys.argv[1]

def dep(logNK, method):
    logN, logK = logNK
    N = 2**logN
    K = 2**logK
    return 'param/rows/' + method + '_' + cpu_gpu + '_' + str(int(N)) + '_' +  str(int(K)) + '.row'

#All pairs of integers up to some sum
def pairs(N):
    Ns = [(i, 7) for i in range(1, N-7+1)]
    Ks = [(7, j) for j in range(1, N-7+1)]
    return [*Ns, *Ks]

maxs = {
    'cpu': {
        'vae' : 10, # 1,000
        'smc' : 10, # 1,000
        'tmc' : 10, # 1,000
        'grt' : 10
        #'t_u' : 10  # 1,000
    },
    'gpu' : {
        'vae' : 26,
        'smc' : 20,
        'tmc' : 17,
        'grt' : 17
    }
}[cpu_gpu]

def deps(method):
    return [dep(logNK, method) for logNK in pairs(maxs[method])]


result_list = [
    #*deps('vae'),
    #*deps('smc'),
    #*deps('tmc'),
    *deps('grt')
]

result_str = " ".join(result_list)
print(result_str)
