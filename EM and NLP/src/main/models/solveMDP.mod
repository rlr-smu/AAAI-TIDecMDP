set S ordered;
set A ordered;
param gamma default 0.7;
param P{i in A, j in S, k in S};
param R{i in S, j in A};
param alpha {i in S};
var x{i in S, j in A};

maximize ER: sum{i in S, j in A} x[i,j]*R[i,j];

s.t. Flow { i in S } :
sum{j in A} x[i,j] - gamma*sum{l in S, k in A}x[l,k]*P[k,l,i] = alpha[i];

s.t. Pos {i in S, j in A} : x[i,j] >= 0;
