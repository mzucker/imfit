import sympy

def check_deriv(name, expr, var, expected):
    actual = expr.diff(var)
    diff = sympy.simplify(expected/actual)
    print 'for', name
    print '  expected:', expected
    print '    actual:', actual
    print '      diff:', diff
    print
    assert( diff == 1 )

x, y, u, v, r, l, s, t, p, h = sympy.symbols('x,y,u,v,r,l,s,t,p,h', real=True)

cr = sympy.cos(r)
sr = sympy.sin(r)

xp = x-u
yp = y-v

b1 = cr*xp + sr*yp
b2 = -sr*xp + cr*yp

db1_du = -cr
db1_dv = -sr
db1_dr = b2

db2_du = sr
db2_dv = -cr
db2_dr = -b1

w = sympy.exp(-b1*b1/(2*s*s) - b2*b2/(2*t*t))

dw_db1 = -w * b1 / (s*s)
dw_db2 = -w * b2 / (t*t)

dw_du = dw_db1 * db1_du + dw_db2 * db2_du
dw_dv = dw_db1 * db1_dv + dw_db2 * db2_dv
dw_dr = dw_db1 * db1_dr + dw_db2 * db2_dr
dw_ds = w * b1 * b1 / (s*s*s)

check_deriv('db1_du', b1, u, db1_du)
check_deriv('db1_dv', b1, v, db1_dv)
check_deriv('db1_dr', b1, r, db1_dr)

check_deriv('db2_du', b2, u, db2_du)
check_deriv('db2_dv', b2, v, db2_dv)
check_deriv('db2_dr', b2, r, db2_dr)

check_deriv('dw_du', w, u, dw_du)
check_deriv('dw_dv', w, v, dw_dv)
check_deriv('dw_dr', w, r, dw_dr)

check_deriv('dw_ds', w, s, dw_ds)
