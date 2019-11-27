param n;
param agent;
param gamma;
set S{1..n} ordered;
set A{1..n} ordered;

set sparseP within {i in 1..n,A[i],S[i],S[i]};
param sparsePVal{sparseP};

param P{i in 1..n, j in A[i], k in S[i], l in S[i]} default 0;
param R{i in 1..n, j in S[i], k in A[i]};
param alpha{i in 1..n, j in S[i]};

#All
set all_numcons ordered;
set all_numprims ordered;
set all_numevents ordered;
param all_primitives{i in all_numprims, j in 1..4};
param all_creward{i in all_numcons};
set all_events{all_numevents} ordered;
set all_cons{all_numcons} ordered;

#Agent Specific
set agent_numcons ordered;
set agent_numprims ordered;
set agent_numevents ordered;
param agent_primitives{i in agent_numprims, j in 1..4};
param agent_creward{i in agent_numcons};
set agent_events{agent_numevents} ordered;
set agent_cons{agent_numcons} ordered;

param Rmax;
param Rmin;
param theta;

param thetahat = theta / (Rmax - Rmin);
param Rcap{j in S[agent], k in A[agent]} = R[agent,j,k] / (Rmax - Rmin);
param crewardcap{i in agent_numcons} = agent_creward[i] / (Rmax - Rmin);

param initialxval;

param x{i in 1..n, j in S[i], k in A[i]} default initialxval;

var xstar{i in S[agent], j in A[agent]};
param z{i in all_numcons, j in all_cons[i]} = sum{k in all_events[j]} x[all_primitives[k,1], all_primitives[k,2], all_primitives[k,3]] * P[all_primitives[k,1], all_primitives[k,3], all_primitives[k,2], all_primitives[k,4]];
var zstar{i in agent_numcons, j in agent_cons[i]};

maximize ER:
( sum{i in S[agent], j in A[agent]} x[agent,i,j] * Rcap[i,j] * log(xstar[i,j])) +
( sum{l in agent_numcons} crewardcap[l] * (prod{m in all_cons[l]} z[l,m]) * sum{o in agent_cons[l]} log(zstar[l,o]) ) +
( sum{l in agent_numcons}  thetahat * sum{o in agent_cons[l]} (1-z[l,o]) * log(1-zstar[l,o]) ) +
( sum{l in agent_numcons}  thetahat * sum{o in agent_cons[l]} (z[l,o]) * log(zstar[l,o]) ) ;

subject to Flow {i in S[agent]}:
sum{j in A[agent]} xstar[i,j] -
gamma * sum{l in S[agent], m in A[agent]} xstar[l,m]*P[agent,m,l,i] == alpha[agent,i];

subject to Positive {i in S[agent], j in A[agent]}: xstar[i,j] >= 0.000001;

subject to defineZStar{i in agent_numcons, j in agent_cons[i]}:
zstar[i,j] == sum{k in agent_events[j]} xstar[agent_primitives[k,2],agent_primitives[k,3]] * P[agent, agent_primitives[k,3], agent_primitives[k,2], agent_primitives[k,4]];
