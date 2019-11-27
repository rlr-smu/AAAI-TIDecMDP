param n;
param gamma;
set S{1..n} ordered;
set A{1..n} ordered;

set sparseP within {i in 1..n,A[i],S[i],S[i]};
param sparsePVal{sparseP};

param P{i in 1..n, j in A[i], k in S[i], l in S[i]} default 0;
param R{i in 1..n, j in S[i], k in A[i]};
param alpha{i in 1..n, j in S[i]};

param numcons;
param numprims;
param numevents;
param primitives{i in 1..numprims, j in 1..4};
param creward{i in 1..numcons};
set events{1..numevents} ordered;
set cons{1..numcons} ordered;

var x{i in 1..n, j in S[i], k in A[i]};
var z{i in 1..numcons, j in cons[i]};

subject to Flow {i in 1..n, j in S[i]}:
sum{k in A[i]} x[i,j,k] -
gamma * sum{l in S[i], m in A[i]} x[i,l,m]*P[i,m,l,j] == alpha[i,j];
subject to Positive {i in 1..n, j in S[i], k in A[i]}: x[i,j,k] >= 0.000001;

subject to defineZ{i in 1..numcons, j in cons[i]}:
#z[i,j] == sum{k in events[j]} ( x[primitives[k,1], primitives[k,2], primitives[k,3]] / sum{acts in A[primitives[k,1]]} x[primitives[k,1],primitives[k,2],acts] ) * P[primitives[k,1], primitives[k,3], primitives[k,2], primitives[k,4]];
z[i,j] == sum{k in events[j]} x[primitives[k,1], primitives[k,2], primitives[k,3]] * P[primitives[k,1], primitives[k,3], primitives[k,2], primitives[k,4]];
