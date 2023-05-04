from numpy import float32, float64


wp = 'float64'
def cst(a):
    if wp == 'float32': 
        a = float32(a)
    elif wp == 'float64': 
        a = float64(a)
    return a


zero    = cst(0.)
one     = cst(1.)
two     = cst(2.)
three   = cst(3.)
four    = cst(4.)
five    = cst(5.)
six     = cst(6.)
eight   = cst(8.)
half    = cst(.5)
fourth  = cst(.25)
threehalf = cst(3./2.)
sixth   = cst(1./6.)
twelth  = cst(1./12.)